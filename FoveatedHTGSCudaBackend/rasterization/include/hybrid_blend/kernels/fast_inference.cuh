#pragma once

#include "helper_math.h"
#include "kernel_utils.cuh"
#include "kernel_utils_aaa.cuh"
#include "hybrid_blend/config.h"
#include <cooperative_groups.h>
#include <cuda_fp16.h>

namespace htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference {

    template<bool aaa_mode>
    __global__ void preprocess_cu(
        const float3* positions,
        const float3* scales,
        const float4* rotations,
        const float* opacities,
        const float3* sh_0,
        const float3* sh_rest,
        uint* primitive_n_touched_tiles,
        uint4* primitive_screen_bounds,
        float4* primitive_VPMT1,
        float4* primitive_VPMT2,
        float4* primitive_VPMT4,
        float4* primitive_MT3,
        float4* primitive_rgba,
        const uint* render_mask_area_table,
        const uint* fovea_mask_area_table,
        const uint n_primitives,
        const uint grid_width,
        const uint grid_height,
        const uint active_sh_bases,
        const uint total_sh_bases,
        const uint2 gaze_position,
        const float width,
        const float height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const float scale_modifier)
    {
        const uint primitive_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (primitive_idx >= n_primitives) return;

        primitive_n_touched_tiles[primitive_idx] = 0;

        const float3 position_world = positions[primitive_idx];
        float opacity = opacities[primitive_idx];
        uint n_touched_tiles;
        uint4 screen_bounds;
        float3 u, v, w;
        float4 VPMT1, VPMT2, VPMT4, MT3;

        // transform and cull
        if constexpr (aaa_mode) {
            // improved transform and cull
            const bool culled = transform_and_cull_aaa(
                scales, rotations,
                position_world,
                n_touched_tiles, screen_bounds, u, v, w, VPMT1, VPMT2, VPMT4, MT3, opacity,
                render_mask_area_table, fovea_mask_area_table,
                primitive_idx, grid_width, grid_height, config::tile_width_large, config::tile_height_large,
                config::foveation_radius_tiles, gaze_position,
                width, height, focal_x, focal_y, center_x, center_y,
                config::min_alpha_threshold, config::min_alpha_threshold_rcp, scale_modifier
            );
            __syncwarp();
            if (culled) return;
        }
        else {
            const float4 M3 = c_M[2];
            float z;
            if (transform_and_cull(
                scales, rotations,
                position_world, M3,
                n_touched_tiles, screen_bounds, u, v, w, VPMT1, VPMT2, VPMT4, z, opacity,
                render_mask_area_table, fovea_mask_area_table,
                primitive_idx, grid_width, grid_height, config::tile_width_large, config::tile_height_large,
                config::foveation_radius_tiles, gaze_position,
                near_plane, far_plane, config::min_alpha_threshold_rcp, scale_modifier
            )) return;
            MT3 = make_float4(dot(make_float3(M3), u), dot(make_float3(M3), v), dot(make_float3(M3), w), z);
        }

        // write intermediate results
        primitive_n_touched_tiles[primitive_idx] = n_touched_tiles;
        primitive_screen_bounds[primitive_idx] = screen_bounds;
        primitive_VPMT1[primitive_idx] = VPMT1;
        primitive_VPMT2[primitive_idx] = VPMT2;
        primitive_VPMT4[primitive_idx] = VPMT4;
        primitive_MT3[primitive_idx] = MT3;

        // compute view-dependent color
        const float3 rgb = convert_sh_to_rgb<false>(
            sh_0,
            sh_rest,
            nullptr,
            position_world,
            n_primitives,
            primitive_idx,
            active_sh_bases,
            total_sh_bases
        );
        primitive_rgba[primitive_idx] = make_float4(rgb, opacity);
    }

    template <int K, bool is_lowres_tile, BackgroundModelType background_model, PeripheryInterpolationMode periphery_mode>
    __global__ void __launch_bounds__(config::block_size_blend) blend_cu(
        const uint* tile_index_map,
        const uint2* tile_instance_ranges,
        const uint* instance_primitive_indices,
        const float4* primitive_VPMT1,
        const float4* primitive_VPMT2,
        const float4* primitive_VPMT4,
        const float4* primitive_MT3,
        const float4* primitive_rgba,
        const float* background_model_data,
        float* image,
        const uint2 gaze_position_tiles,
        const uint tile_offset,
        const uint width,
        const uint height,
        const uint grid_width,
        const bool output_chw)
    {
        const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        const uint group_index = block.group_index().x + tile_offset;
        const uint true_group_index = tile_index_map[group_index];
        const uint large_tile_index = true_group_index / config::num_small_tiles_per_large_tile;
        const uint subtile_index = true_group_index % config::num_small_tiles_per_large_tile;
        const dim3 large_tile_index_2d(large_tile_index % grid_width, large_tile_index / grid_width, 0);
        const dim3 subtile_index_2d(subtile_index % config::tile_stride_x, subtile_index / config::tile_stride_x, 0);

        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        const uint2 pixel_coords = make_uint2(
            large_tile_index_2d.x * config::tile_width_large + subtile_index_2d.x * config::tile_width_small + thread_index.x * (is_lowres_tile ? config::tile_stride_x : 1),
            large_tile_index_2d.y * config::tile_height_large + subtile_index_2d.y * config::tile_height_small + thread_index.y * (is_lowres_tile ? config::tile_stride_y : 1)
        );
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;

        constexpr const float pixel_offset = (is_lowres_tile && periphery_mode == PeripheryInterpolationMode::NEAREST) ? 0.5f : 0.0f;
        const float pixel_x = __uint2float_rn(pixel_coords.x) + pixel_offset;
        const float pixel_y = __uint2float_rn(pixel_coords.y) + pixel_offset;
        // setup shared memory
        __shared__ float4 collected_VPMT1[config::block_size_blend], collected_VPMT2[config::block_size_blend], collected_VPMT4[config::block_size_blend], collected_MT3[config::block_size_blend];
        __shared__ float3 collected_rgb[config::block_size_blend];
        __shared__ float collected_opacity[config::block_size_blend];
        // initialize local storage
        float transmittance_tail = 1.0f;
        float4 rgba_premultiplied_tail = make_float4(0.0f);
        __half2 rgbas_premultiplied_core_rg[K];
        __half2 rgbas_premultiplied_core_ba[K];
        float depths_core[K];
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            rgbas_premultiplied_core_rg[i]= {0};
            rgbas_premultiplied_core_ba[i]= {0};
            depths_core[i] = __FLT_MAX__;
        }
        // collaborative loading and processing
        const uint2 tile_range = tile_instance_ranges[true_group_index];
        for (int n_points_remaining = tile_range.y - tile_range.x, current_fetch_idx = tile_range.x + thread_rank; n_points_remaining > 0; n_points_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
            block.sync();
            if (current_fetch_idx < tile_range.y) {
                const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
                collected_VPMT1[thread_rank] = primitive_VPMT1[primitive_idx];
                collected_VPMT2[thread_rank] = primitive_VPMT2[primitive_idx];
                collected_VPMT4[thread_rank] = primitive_VPMT4[primitive_idx];
                collected_MT3[thread_rank] = primitive_MT3[primitive_idx];
                const float4 rgba = primitive_rgba[primitive_idx];
                collected_rgb[thread_rank] = make_float3(rgba.x, rgba.y, rgba.z);
                collected_opacity[thread_rank] = rgba.w;
            }
            block.sync();
            if (inside) {
                const int current_batch_size = min(config::block_size_blend, n_points_remaining);
                for (int j = 0; j < current_batch_size; ++j) {
                    const float4 VPMT1 = collected_VPMT1[j];
                    const float4 VPMT2 = collected_VPMT2[j];
                    const float4 VPMT4 = collected_VPMT4[j];
                    const float4 plane_x_diag = VPMT1 - VPMT4 * pixel_x;
                    const float4 plane_y_diag = VPMT2 - VPMT4 * pixel_y;
                    const float3 plane_x_diag_normal = make_float3(plane_x_diag);
                    const float3 plane_y_diag_normal = make_float3(plane_y_diag);
                    const float3 m = plane_x_diag.w * plane_y_diag_normal - plane_x_diag_normal * plane_y_diag.w;
                    const float3 d = cross(plane_x_diag_normal, plane_y_diag_normal);
                    const float numerator_rho2 = dot(m, m);
                    const float denominator = dot(d, d);
                    if (numerator_rho2 > config::max_cutoff_sq * denominator) continue; // considering opacity requires log/sqrt -> slower
                    const float denominator_rcp = 1.0f / denominator;
                    const float3 eval_point_diag = cross(d, m) * denominator_rcp;
                    const float4 MT3 = collected_MT3[j];
                    float depth = dot(make_float3(MT3), eval_point_diag) + MT3.w;
                    const float G = expf(-0.5f * numerator_rho2 * denominator_rcp);
                    const float alpha = fminf(collected_opacity[j] * G, config::max_fragment_alpha);
                    if (alpha < config::min_alpha_threshold) continue;

                    const float3 rgb = collected_rgb[j];
                    __half2 rgba_premultiplied_rg = __float22half2_rn(make_float2(rgb.x * alpha, rgb.y * alpha));
                    __half2 rgba_premultiplied_ba = __float22half2_rn(make_float2(rgb.z * alpha, alpha));

                    if (depth < depths_core[K - 1] && alpha >= config::min_alpha_threshold_core) {
                        #pragma unroll
                        for (int core_idx = 0; core_idx < K; ++core_idx) {
                            if (depth < depths_core[core_idx]) {
                                swap(depth, depths_core[core_idx]);
                                swap(rgba_premultiplied_rg, rgbas_premultiplied_core_rg[core_idx]);
                                swap(rgba_premultiplied_ba, rgbas_premultiplied_core_ba[core_idx]);
                            }
                        }
                    }
                    const float4 primitive_rgba_premultiplied_tail = make_float4(__half2float(rgba_premultiplied_rg.x), __half2float(rgba_premultiplied_rg.y), __half2float(rgba_premultiplied_ba.x), __half2float(rgba_premultiplied_ba.y));
                    const float primitive_transmittance_tail = 1.0f - __half2float(rgba_premultiplied_ba.y);
                    rgba_premultiplied_tail += primitive_rgba_premultiplied_tail;
                    transmittance_tail *= primitive_transmittance_tail;
                }
            }
        }
        if (inside) {
            // blend core
            float3 rgb_pixel = make_float3(0.0f);
            float transmittance_core = 1.0f;
            float transmittance_core_blended = 1.0f;
            bool done = false;
            bool done_blended = false;
            #pragma unroll
            for (int core_idx = 0; core_idx < K && !done; ++core_idx) {
                const float2 rgba_premultiplied_rg = __half22float2(rgbas_premultiplied_core_rg[core_idx]);
                const float2 rgba_premultiplied_ba = __half22float2(rgbas_premultiplied_core_ba[core_idx]);
                const float3 rgb_premultiplied = make_float3(rgba_premultiplied_rg.x, rgba_premultiplied_rg.y, rgba_premultiplied_ba.x);
                rgb_pixel += transmittance_core * rgb_premultiplied;
                transmittance_core *= 1.0f - rgba_premultiplied_ba.y;
                if (transmittance_core < config::transmittance_threshold) done = true;
            }
            // blend tail
            if (!done && rgba_premultiplied_tail.w >= config::min_alpha_threshold) {
                const float weight_tail = transmittance_core * (1.0f - transmittance_tail);
                rgb_pixel += weight_tail * (1.0f / rgba_premultiplied_tail.w) * make_float3(rgba_premultiplied_tail);
            }
            // blend background model
            if constexpr (background_model == BackgroundModelType::SH) {
                const float weight_background = transmittance_core * transmittance_tail;
                if (weight_background >= config::transmittance_threshold) {
                    rgb_pixel += weight_background * eval_sh_background_model(pixel_x, pixel_y);
                }
            } else if constexpr (background_model == BackgroundModelType::TEXTURE) {
                const float weight_background = transmittance_core * transmittance_tail;
                if (weight_background >= config::transmittance_threshold) {
                    rgb_pixel += weight_background * eval_tex_background_model<config::environment_map_width, config::environment_map_height>(
                        pixel_x, pixel_y, background_model_data
                    );
                }
            }
        }
    }

    __global__ void __launch_bounds__(config::block_size_blur) interpolate_missing(
        float* image,
        const uint width,
        const uint height,
        const uint grid_width,
        const uint2 gaze_position_tiles,
        const bool output_chw
    ) {
        const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        const dim3 group_index = block.group_index();
        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        const uint large_tile_index = group_index.y / config::tile_stride_y * grid_width + group_index.x / config::tile_stride_x;
        const bool is_fovea_tile = is_in_fovea<config::foveation_radius_tiles>(large_tile_index, grid_width, gaze_position_tiles);
        if (is_fovea_tile) return;

        const uint2 pixel_coords = make_uint2(
            group_index.x * config::tile_width_small + thread_index.x,
            group_index.y * config::tile_height_small + thread_index.y
        );
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        if (!inside) return;

        const uint2 pixel_stride_coords = make_uint2(
            thread_index.x % config::tile_stride_x,
            thread_index.y % config::tile_stride_y
        );
        if (pixel_stride_coords.x == 0 && pixel_stride_coords.y == 0) {
            // Top left pixel is already set
            return;
        }

        const float4 bilinear_factors = make_float4(
            (1.0 - float(thread_index.x % config::tile_stride_x) / float(config::tile_stride_x))
            * (1.0 - float(thread_index.y % config::tile_stride_y) / float(config::tile_stride_y)),  // Top Left

            (float(thread_index.x % config::tile_stride_x) / float(config::tile_stride_x))
            * (1.0 - float(thread_index.y % config::tile_stride_y) / float(config::tile_stride_y)),  // Top Right

            (1.0 - float(thread_index.x % config::tile_stride_x) / float(config::tile_stride_x))
            * (float(thread_index.y % config::tile_stride_y) / float(config::tile_stride_y)),  // Bottom Left

            (float(thread_index.x % config::tile_stride_x) / float(config::tile_stride_x))
            * (float(thread_index.y % config::tile_stride_y) / float(config::tile_stride_y))  // Bottom Right
        );
        const uint2 top_left_pixel_coords = make_uint2(
            pixel_coords.x - pixel_stride_coords.x,
            pixel_coords.y - pixel_stride_coords.y
        );
        const uint2 sample_coords[4] = {
            top_left_pixel_coords,
            make_uint2(min(top_left_pixel_coords.x + config::tile_stride_x, width - 1), top_left_pixel_coords.y),
            make_uint2(top_left_pixel_coords.x, min(top_left_pixel_coords.y + config::tile_stride_y, height - 1)),
            make_uint2(min(top_left_pixel_coords.x + config::tile_stride_x, width - 1), min(top_left_pixel_coords.y + config::tile_stride_y, height - 1))
        };

        const int pixel_idx = width * pixel_coords.y + pixel_coords.x;
        const int sample_indices[4] = {
            width * sample_coords[0].y + sample_coords[0].x,
            width * sample_coords[1].y + sample_coords[1].x,
            width * sample_coords[2].y + sample_coords[2].x,
            width * sample_coords[3].y + sample_coords[3].x
        };

        if (output_chw) {
            const int n_pixels = width * height;
            image[pixel_idx] = bilinear_factors.x * image[sample_indices[0]] +
                               bilinear_factors.y * image[sample_indices[1]] +
                               bilinear_factors.z * image[sample_indices[2]] +
                               bilinear_factors.w * image[sample_indices[3]];
            image[n_pixels + pixel_idx] = bilinear_factors.x * image[n_pixels + sample_indices[0]] +
                                          bilinear_factors.y * image[n_pixels + sample_indices[1]] +
                                          bilinear_factors.z * image[n_pixels + sample_indices[2]] +
                                          bilinear_factors.w * image[n_pixels + sample_indices[3]];
            image[2 * n_pixels + pixel_idx] = bilinear_factors.x * image[2 * n_pixels + sample_indices[0]] +
                                              bilinear_factors.y * image[2 * n_pixels + sample_indices[1]] +
                                              bilinear_factors.z * image[2 * n_pixels + sample_indices[2]] +
                                              bilinear_factors.w * image[2 * n_pixels + sample_indices[3]];
        } else {
            const int base_idx = 3 * pixel_idx;
            image[base_idx] = bilinear_factors.x * image[3 * sample_indices[0]] +
                              bilinear_factors.y * image[3 * sample_indices[1]] +
                              bilinear_factors.z * image[3 * sample_indices[2]] +
                              bilinear_factors.w * image[3 * sample_indices[3]];
            image[base_idx + 1] = bilinear_factors.x * image[3 * sample_indices[0] + 1] +
                                  bilinear_factors.y * image[3 * sample_indices[1] + 1] +
                                  bilinear_factors.z * image[3 * sample_indices[2] + 1] +
                                  bilinear_factors.w * image[3 * sample_indices[3] + 1];
            image[base_idx + 2] = bilinear_factors.x * image[3 * sample_indices[0] + 2] +
                                  bilinear_factors.y * image[3 * sample_indices[1] + 2] +
                                  bilinear_factors.z * image[3 * sample_indices[2] + 2] +
                                  bilinear_factors.w * image[3 * sample_indices[3] + 2];
        }
    }

    __global__ void __launch_bounds__(config::block_size_blur) blur_cu(
        float* image,
        float* image_blurred,
        const uint* tile_index_map,
        const uint tile_offset,
        const uint width,
        const uint height,
        const uint grid_width,
        const bool output_chw
    ) {
        constexpr float gaussian_kernel_factors[3][3] = {
            {1.0f, 2.0f, 1.0f},
            {2.0f, 4.0f, 2.0f},
            {1.0f, 2.0f, 1.0f}
        };
        constexpr float gaussian_factor = 1.0f / 16.0f;

        const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        const uint group_index = block.group_index().x + tile_offset;
        const uint true_group_index = tile_index_map[group_index];
        const uint large_tile_index = true_group_index / config::num_small_tiles_per_large_tile;
        const dim3 large_tile_index_2d(large_tile_index % grid_width, large_tile_index / grid_width, 0);
        const dim3 thread_index = block.thread_index();
        const uint pixel_offset_x = block.group_index().y % config::tile_stride_x;
        const uint pixel_offset_y = block.group_index().y / config::tile_stride_x;

        const uint2 pixel_coords = make_uint2(
            large_tile_index_2d.x * config::tile_width_large + thread_index.x * config::tile_stride_x + pixel_offset_x,
            large_tile_index_2d.y * config::tile_width_large + thread_index.y * config::tile_stride_y + pixel_offset_y
        );
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        if (!inside) return;
        const int pixel_idx = width * pixel_coords.y + pixel_coords.x;

        float3 average_rgb_pixel = make_float3(0.0f);
        #pragma unroll
        for (int x_off = -1; x_off < 2; x_off++) {
            #pragma unroll
            for (int y_off = -1; y_off < 2; y_off++) {
                const int sample_x = min(max(int(pixel_coords.x) + x_off, 0), int(width) - 1);
                const int sample_y = min(max(int(pixel_coords.y) + y_off, 0), int(height) - 1);
                const int sample_idx = width * sample_y + sample_x;

                if (output_chw) {
                    const int n_pixels = width * height;
                    average_rgb_pixel.x += image[sample_idx] * gaussian_kernel_factors[x_off + 1][y_off + 1];
                    average_rgb_pixel.y += image[n_pixels + sample_idx] * gaussian_kernel_factors[x_off + 1][y_off + 1];
                    average_rgb_pixel.z += image[2 * n_pixels + sample_idx] * gaussian_kernel_factors[x_off + 1][y_off + 1];
                } else {
                    const int base_idx = 3 * sample_idx;
                    average_rgb_pixel.x += image[base_idx] * gaussian_kernel_factors[x_off + 1][y_off + 1];
                    average_rgb_pixel.y += image[base_idx + 1] * gaussian_kernel_factors[x_off + 1][y_off + 1];
                    average_rgb_pixel.z += image[base_idx + 2] * gaussian_kernel_factors[x_off + 1][y_off + 1];
                }
            }
        }

        if (output_chw) {
            const int n_pixels = width * height;
            image_blurred[pixel_idx] = __saturatef(average_rgb_pixel.x * gaussian_factor);
            image_blurred[n_pixels + pixel_idx] = __saturatef(average_rgb_pixel.y * gaussian_factor);
            image_blurred[2 * n_pixels + pixel_idx] = __saturatef(average_rgb_pixel.z * gaussian_factor);
        } else {
            const int base_idx = 3 * pixel_idx;
            image_blurred[base_idx] = __saturatef(average_rgb_pixel.x * gaussian_factor);
            image_blurred[base_idx + 1] = __saturatef(average_rgb_pixel.y * gaussian_factor);
            image_blurred[base_idx + 2] = __saturatef(average_rgb_pixel.z * gaussian_factor);
        }
    }

    __forceinline__ __device__ float3 sample_rgb(
        const float* image,
        const int x,
        const int y,
        const int width,
        const int height,
        const bool output_chw
    ) {
        const int sample_x = clamp(x, 0, width - 1);
        const int sample_y = clamp(y, 0, height - 1);
        const uint sample_idx = width * sample_y + sample_x;

        if (output_chw) {
            const uint n_pixels = width * height;
            return make_float3(
                image[sample_idx],
                image[n_pixels + sample_idx],
                image[2 * n_pixels + sample_idx]
            );
        } else {
            const uint base_idx = 3 * sample_idx;
            return make_float3(
                image[base_idx],
                image[base_idx + 1],
                image[base_idx + 2]
            );
        }
    }

    __global__ void __launch_bounds__(config::block_size_blur) interpolate_and_blur(
        float* image_blurred,
        const float* image,
        const uint* tile_index_map,
        const uint tile_offset,
        const uint width,
        const uint height,
        const uint grid_width,
        const bool output_chw
    ) {
        // Fused bilinear interpolation + gaussian blur kernel
        // Because the workload differs based on the subpixel (in each 2x2 group of pixels) differs,
        //   we launch four blocks (differentiated by y/z block index) per tile, each responsible
        //   for one of the subpixels
        // TODO: Compare the performance with 4 (templated) separate kernels
        const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        const uint group_index = block.group_index().x + tile_offset;
        const uint true_group_index = tile_index_map[group_index];
        const uint large_tile_index = true_group_index / config::num_small_tiles_per_large_tile;
        const dim3 large_tile_index_2d(large_tile_index % grid_width, large_tile_index / grid_width, 0);

        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        int pixel_x = large_tile_index_2d.x * config::tile_width_large + thread_index.x * config::tile_stride_x + block.group_index().y;
        int pixel_y = large_tile_index_2d.y * config::tile_height_large + thread_index.y * config::tile_stride_y + block.group_index().z;
        const bool inside = pixel_x < width && pixel_y < height;
        if (!inside) return;
        const int pixel_idx = width * pixel_y + pixel_x;

        // TODO: The following assumes pixel stride of 2, remove unnecessary generalizations in the code before code release
        // The method here would work just the same with higher strides, but would require different factors
        float3 blurred_rgb = make_float3(0.0f);
        float blur_factor;
        if (block.group_index().y == 0 && block.group_index().z == 0) {
            // Top left pixel, factors:
            //   [1.0   0.0   6.0  0.0  1.0]
            //   [0.0   0.0   0.0  0.0  0.0]
            //   [6.0   0.0  36.0  0.0  6.0]
            //   [0.0   0.0   0.0  0.0  0.0]
            //   [1.0   0.0   6.0  0.0  1.0]
            blur_factor = 1.0f / 64.0f;
            blurred_rgb += sample_rgb(image, pixel_x - 2, pixel_y - 2, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x,     pixel_y - 2, width, height, output_chw) *  6.0f;
            blurred_rgb += sample_rgb(image, pixel_x + 2, pixel_y - 2, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x - 2, pixel_y,     width, height, output_chw) *  6.0f;
            blurred_rgb += sample_rgb(image, pixel_x,     pixel_y,     width, height, output_chw) * 36.0f;
            blurred_rgb += sample_rgb(image, pixel_x + 2, pixel_y,     width, height, output_chw) *  6.0f;
            blurred_rgb += sample_rgb(image, pixel_x - 2, pixel_y + 2, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x,     pixel_y + 2, width, height, output_chw) *  6.0f;
            blurred_rgb += sample_rgb(image, pixel_x + 2, pixel_y + 2, width, height, output_chw);
        } else if (block.group_index().y == 0 && block.group_index().z == 1) {
            // Bottom left pixel, factors:
            //   [1.0  0.0  6.0  0.0  1.0]
            //   [0.0  0.0  0.0  0.0  0.0]
            //   [1.0  0.0  6.0  0.0  1.0]
            blur_factor = 1.0f / 16.0f;
            blurred_rgb += sample_rgb(image, pixel_x - 2, pixel_y - 1, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x,     pixel_y - 1, width, height, output_chw) *  6.0f;
            blurred_rgb += sample_rgb(image, pixel_x + 2, pixel_y - 1, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x - 2, pixel_y + 1, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x,     pixel_y + 1, width, height, output_chw) *  6.0f;
            blurred_rgb += sample_rgb(image, pixel_x + 2, pixel_y + 1, width, height, output_chw);
        } else if (block.group_index().y == 1 && block.group_index().z == 0) {
            // Top right pixel, factors:
            //   [1.0  0.0  1.0]
            //   [0.0  0.0  0.0]
            //   [6.0  0.0  6.0]
            //   [0.0  0.0  0.0]
            //   [1.0  0.0  1.0]
            blur_factor = 1.0f / 16.0f;
            blurred_rgb += sample_rgb(image, pixel_x - 1, pixel_y - 2, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x + 1, pixel_y - 2, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x - 1, pixel_y,     width, height, output_chw) * 6.0f;
            blurred_rgb += sample_rgb(image, pixel_x + 1, pixel_y,     width, height, output_chw) * 6.0f;
            blurred_rgb += sample_rgb(image, pixel_x - 1, pixel_y + 2, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x + 1, pixel_y + 2, width, height, output_chw);
        } else if (block.group_index().y == 1 && block.group_index().z == 1) {
            // Bottom right pixel, factors:
            //   [1.0  0.0  1.0]
            //   [0.0  0.0  0.0]
            //   [1.0  0.0  1.0]
            blur_factor = 1.0f / 4.0f;
            blurred_rgb += sample_rgb(image, pixel_x - 1, pixel_y - 1, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x + 1, pixel_y - 1, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x - 1, pixel_y + 1, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x + 1, pixel_y + 1, width, height, output_chw);
        }

        if (output_chw) {
            const int n_pixels = width * height;
            image_blurred[pixel_idx] = __saturatef(blurred_rgb.x * blur_factor);
            image_blurred[n_pixels + pixel_idx] = __saturatef(blurred_rgb.y * blur_factor);
            image_blurred[2 * n_pixels + pixel_idx] = __saturatef(blurred_rgb.z * blur_factor);
        } else {
            const int base_idx = 3 * pixel_idx;
            image_blurred[base_idx] = __saturatef(blurred_rgb.x * blur_factor);
            image_blurred[base_idx + 1] = __saturatef(blurred_rgb.y * blur_factor);
            image_blurred[base_idx + 2] = __saturatef(blurred_rgb.z * blur_factor);
        }
    }


    __global__ void __launch_bounds__(config::block_size_blur_blended) interpolate_and_blur_blended(
        float* image_blurred,
        const float* image,
        const uint* tile_index_map,
        const uint tile_offset,
        const uint width,
        const uint height,
        const uint grid_width,
        const uint2 gaze_position_tiles,
        const bool output_chw
    ) {
        // Fused bilinear interpolation + gaussian blur kernel; with output blended with existing pixels (for blended tiles)
        // Because the workload differs based on the subpixel (in each 2x2 group of pixels) differs,
        //   we launch four blocks (differentiated by y/z block index) per tile, each responsible
        //   for one of the subpixels
        // TODO: Compare the performance with 4 (templated) separate kernels
        const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        const uint group_index = block.group_index().x + tile_offset;
        const uint true_group_index = tile_index_map[group_index];
        const uint large_tile_index = true_group_index / config::num_small_tiles_per_large_tile;
        const uint subtile_index = true_group_index % config::num_small_tiles_per_large_tile;
        const dim3 large_tile_index_2d(large_tile_index % grid_width, large_tile_index / grid_width, 0);
        const dim3 subtile_index_2d(subtile_index % config::tile_stride_x, subtile_index / config::tile_stride_x, 0);

        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        int pixel_x = large_tile_index_2d.x * config::tile_width_large + subtile_index_2d.x * config::tile_width_small + thread_index.x * config::tile_stride_x + block.group_index().y;
        int pixel_y = large_tile_index_2d.y * config::tile_height_large + subtile_index_2d.y * config::tile_height_small + thread_index.y * config::tile_stride_y + block.group_index().z;
        const bool inside = pixel_x < width && pixel_y < height;
        if (!inside) return;
        const int pixel_idx = width * pixel_y + pixel_x;

        // Determine blending factor
        const float2 dist_from_gaze = c_gaze_position_cuda - make_float2(pixel_x, pixel_y);
        const float blend_factor = clamp((length(dist_from_gaze) - config::blend_radius) / config::blend_width, 0.0f, 1.0f);

        // TODO: The following assumes pixel stride of 2, remove unnecessary generalizations in the code before code release
        // The method here would work just the same with higher strides, but would require different factors
        float3 blurred_rgb = make_float3(0.0f);
        float3 center_rgb = sample_rgb(image, pixel_x, pixel_y, width, height, output_chw);
        float blur_factor;
        if (block.group_index().y == 0 && block.group_index().z == 0) {
            // Top left pixel, factors:
            //   [1.0   0.0   6.0  0.0  1.0]
            //   [0.0   0.0   0.0  0.0  0.0]
            //   [6.0   0.0  36.0  0.0  6.0]
            //   [0.0   0.0   0.0  0.0  0.0]
            //   [1.0   0.0   6.0  0.0  1.0]
            blur_factor = 1.0f / 64.0f;
            blurred_rgb += sample_rgb(image, pixel_x - 2, pixel_y - 2, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x,     pixel_y - 2, width, height, output_chw) *  6.0f;
            blurred_rgb += sample_rgb(image, pixel_x + 2, pixel_y - 2, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x - 2, pixel_y,     width, height, output_chw) *  6.0f;
            blurred_rgb += sample_rgb(image, pixel_x,     pixel_y,     width, height, output_chw) * 36.0f;
            blurred_rgb += sample_rgb(image, pixel_x + 2, pixel_y,     width, height, output_chw) *  6.0f;
            blurred_rgb += sample_rgb(image, pixel_x - 2, pixel_y + 2, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x,     pixel_y + 2, width, height, output_chw) *  6.0f;
            blurred_rgb += sample_rgb(image, pixel_x + 2, pixel_y + 2, width, height, output_chw);
        } else if (block.group_index().y == 1 && block.group_index().z == 0) {
            // Top right pixel, factors:
            //   [1.0  0.0  6.0  0.0  1.0]
            //   [0.0  0.0  0.0  0.0  0.0]
            //   [1.0  0.0  6.0  0.0  1.0]
            blur_factor = 1.0f / 16.0f;
            blurred_rgb += sample_rgb(image, pixel_x - 2, pixel_y - 1, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x,     pixel_y - 1, width, height, output_chw) *  6.0f;
            blurred_rgb += sample_rgb(image, pixel_x + 2, pixel_y - 1, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x - 2, pixel_y + 1, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x,     pixel_y + 1, width, height, output_chw) *  6.0f;
            blurred_rgb += sample_rgb(image, pixel_x + 2, pixel_y + 1, width, height, output_chw);
        } else if (block.group_index().y == 0 && block.group_index().z == 1) {
            // Bottom left pixel, factors:
            //   [1.0  0.0  1.0]
            //   [0.0  0.0  0.0]
            //   [6.0  0.0  6.0]
            //   [0.0  0.0  0.0]
            //   [1.0  0.0  1.0]
            blur_factor = 1.0f / 16.0f;
            blurred_rgb += sample_rgb(image, pixel_x - 1, pixel_y - 2, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x + 1, pixel_y - 2, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x - 1, pixel_y,     width, height, output_chw) * 6.0f;
            blurred_rgb += sample_rgb(image, pixel_x + 1, pixel_y,     width, height, output_chw) * 6.0f;
            blurred_rgb += sample_rgb(image, pixel_x - 1, pixel_y + 2, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x + 1, pixel_y + 2, width, height, output_chw);
        } else if (block.group_index().y == 1 && block.group_index().z == 1) {
            // Bottom right pixel, factors:
            //   [1.0  0.0  1.0]
            //   [0.0  0.0  0.0]
            //   [1.0  0.0  1.0]
            blur_factor = 1.0f / 4.0f;
            blurred_rgb += sample_rgb(image, pixel_x - 1, pixel_y - 1, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x + 1, pixel_y - 1, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x - 1, pixel_y + 1, width, height, output_chw);
            blurred_rgb += sample_rgb(image, pixel_x + 1, pixel_y + 1, width, height, output_chw);
        }

        if (output_chw) {
            const int n_pixels = width * height;
            image_blurred[pixel_idx] = __saturatef(center_rgb.x * (1.0f - blend_factor) + blurred_rgb.x * blur_factor * blend_factor);
            image_blurred[n_pixels + pixel_idx] = __saturatef(center_rgb.y * (1.0f - blend_factor) + blurred_rgb.y * blur_factor * blend_factor);
            image_blurred[2 * n_pixels + pixel_idx] = __saturatef(center_rgb.z * (1.0f - blend_factor) + blurred_rgb.z * blur_factor * blend_factor);
        } else {
            const int base_idx = 3 * pixel_idx;
            image_blurred[base_idx] = __saturatef(center_rgb.x * (1.0f - blend_factor) + blurred_rgb.x * blur_factor * blend_factor);
            image_blurred[base_idx + 1] = __saturatef(center_rgb.y * (1.0f - blend_factor) + blurred_rgb.y * blur_factor * blend_factor);
            image_blurred[base_idx + 2] = __saturatef(center_rgb.z * (1.0f - blend_factor) + blurred_rgb.z * blur_factor * blend_factor);
        }
    }

    __global__ void __launch_bounds__(config::block_size_blur) copy_pixels(
        float* src,
        float* dst,
        const uint* tile_index_map,
        const uint tile_offset,
        const uint width,
        const uint height,
        const uint grid_width,
        const bool output_chw
    ) {
        const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        const uint group_index = block.group_index().x + tile_offset;
        const uint true_group_index = tile_index_map[group_index];
        const uint large_tile_index = true_group_index / config::num_small_tiles_per_large_tile;
        const dim3 large_tile_index_2d(large_tile_index % grid_width, large_tile_index / grid_width, 0);
        const dim3 thread_index = block.thread_index();
        const uint pixel_offset_x = block.group_index().y % config::tile_stride_x;
        const uint pixel_offset_y = block.group_index().y / config::tile_stride_x;

        const uint2 pixel_coords = make_uint2(
            large_tile_index_2d.x * config::tile_width_large + thread_index.x * config::tile_stride_x + pixel_offset_x,
            large_tile_index_2d.y * config::tile_width_large + thread_index.y * config::tile_stride_y + pixel_offset_y
        );
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        if (!inside) return;
        const int pixel_idx = width * pixel_coords.y + pixel_coords.x;

        if (output_chw) {
            const int n_pixels = width * height;
            dst[pixel_idx] = src[pixel_idx];
            dst[n_pixels + pixel_idx] = src[n_pixels + pixel_idx];
            dst[2 * n_pixels + pixel_idx] = src[2 * n_pixels + pixel_idx];
        } else {
            const int base_idx = 3 * pixel_idx;
            dst[base_idx] = src[base_idx];
            dst[base_idx + 1] = src[base_idx + 1];
            dst[base_idx + 2] = src[base_idx + 2];
        }
    }

    __global__ void __launch_bounds__(config::gaze_visualization_size) visualize_gaze(
        float* image,
        const uint width,
        const uint height,
        const bool output_chw
    ) {
        const dim3 group_index = cooperative_groups::this_thread_block().group_index();
        const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        const dim3 thread_index = block.thread_index();
        const int x_off = group_index.x * config::tile_width_small + thread_index.x - config::gaze_visualization_width / 2;
        const int y_off = group_index.y * config::tile_width_small + thread_index.y - config::gaze_visualization_width / 2;

        if constexpr (config::gaze_visualization_circular) {
            if (x_off * x_off + y_off * y_off > (config::gaze_visualization_width / 2) * (config::gaze_visualization_width / 2)) return;
        }

        const int x = static_cast<int>(c_gaze_position_cuda.x) + x_off;
        const int y = static_cast<int>(c_gaze_position_cuda.y) + y_off;
        if (x < 0 || x >= width || y < 0 || y >= height) return;

        const int pixel_idx = width * y + x;
        if (output_chw) {
            const int n_pixels = width * height;
            image[pixel_idx] = 1.0f;
            image[n_pixels + pixel_idx] = 0.0f;
            image[2 * n_pixels + pixel_idx] = 0.0f;
        } else {
            const int base_idx = 3 * pixel_idx;
            image[base_idx] = 1.0f;
            image[base_idx + 1] = 0.0f;
            image[base_idx + 2] = 0.0f;
        }
    }
}
