#pragma once

#include "helper_math.h"
#include "kernel_utils.cuh"
#include "hybrid_blend/config.h"
#include <cooperative_groups.h>

namespace htgs::rasterization::hybrid_blend::kernels::inference {

    template<bool anti_aliasing>
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
        const float focal_x,
        const float focal_y,
        const float near_plane,
        const float far_plane,
        const float scale_modifier)
    {
        const uint primitive_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (primitive_idx >= n_primitives) return;

        primitive_n_touched_tiles[primitive_idx] = 0;

        // transform and cull
        float opacity = opacities[primitive_idx];
        const float3 position_world = positions[primitive_idx];
        const float4 M3 = c_M3;
        uint n_touched_tiles;
        uint4 screen_bounds;
        float3 u, v, w;
        float4 VPMT1, VPMT2, VPMT4;
        float z;
        if (transform_and_cull<anti_aliasing>(
            scales, rotations,
            position_world, M3,
            n_touched_tiles, screen_bounds, u, v, w, VPMT1, VPMT2, VPMT4, z, opacity,
            render_mask_area_table, fovea_mask_area_table,
            primitive_idx, grid_width, grid_height, config::tile_width_large, config::tile_height_large,
            config::foveation_radius_tiles, gaze_position, focal_x, focal_y,
            near_plane, far_plane, config::min_alpha_threshold_rcp, scale_modifier, config::aa_kernel_size
        )) return;

        // write intermediate results
        primitive_n_touched_tiles[primitive_idx] = n_touched_tiles;
        primitive_screen_bounds[primitive_idx] = screen_bounds;
        primitive_VPMT1[primitive_idx] = VPMT1;
        primitive_VPMT2[primitive_idx] = VPMT2;
        primitive_VPMT4[primitive_idx] = VPMT4;
        primitive_MT3[primitive_idx] = make_float4(dot(make_float3(M3), u), dot(make_float3(M3), v), dot(make_float3(M3), w), z);

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

    template <int K, bool is_lowres_tile, bool is_blended_tile, PeripheryInterpolationMode periphery_mode>
    __global__ void __launch_bounds__(config::block_size_blend) blend_cu(
        const uint* tile_index_map,
        const uint2* tile_instance_ranges,
        const uint* instance_primitive_indices,
        const float4* primitive_VPMT1,
        const float4* primitive_VPMT2,
        const float4* primitive_VPMT4,
        const float4* primitive_MT3,
        const float4* primitive_rgba,
        float* image,
        float* depths,
        const uint2 gaze_position_tiles,
        const uint tile_offset,
        const uint width,
        const uint height,
        const uint grid_width,
        const bool output_chw,
        const bool use_median_depth)
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
        float depth_premultiplied_tail;
        if (!use_median_depth) depth_premultiplied_tail = 0.0f;
        float4 rgbas_premultiplied_core[K];
        float depths_core[K];
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            rgbas_premultiplied_core[i] = make_float4(0.0f);
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
                    float4 rgba_premultiplied = make_float4(rgb.x * alpha, rgb.y * alpha, rgb.z * alpha, alpha);
                    if (depth < depths_core[K - 1] && alpha >= config::min_alpha_threshold_core) {
                        #pragma unroll
                        for (int core_idx = 0; core_idx < K; ++core_idx) {
                            if (depth < depths_core[core_idx]) {
                                swap(depth, depths_core[core_idx]);
                                swap(rgba_premultiplied, rgbas_premultiplied_core[core_idx]);
                            }
                        }
                    }
                    rgba_premultiplied_tail += rgba_premultiplied;
                    if (!use_median_depth) depth_premultiplied_tail += depth * rgba_premultiplied.w;
                    transmittance_tail *= 1.0f - rgba_premultiplied.w;
                }
            }
        }
        if (inside) {
            // blend core
            float3 rgb_pixel = make_float3(0.0f);
            float depth_pixel = 0.0f;
            float transmittance_core = 1.0f;
            bool done = false;
            #pragma unroll
            for (int core_idx = 0; core_idx < K && !done; ++core_idx) {
                const float4 rgba_premultiplied = rgbas_premultiplied_core[core_idx];

                rgb_pixel += transmittance_core * make_float3(rgba_premultiplied);

                const float depth = depths_core[core_idx];
                if (use_median_depth) depth_pixel = (transmittance_core > 0.5f && depth < __FLT_MAX__) ? depth : depth_pixel;
                else depth_pixel += transmittance_core * rgba_premultiplied.w * depth;

                transmittance_core *= 1.0f - rgba_premultiplied.w;
                if (transmittance_core < config::transmittance_threshold) done = true;
            }
            float total_alpha;
            if (!use_median_depth) total_alpha = 1.0f - transmittance_core;
            // blend tail
            if (!done && rgba_premultiplied_tail.w >= config::min_alpha_threshold) {
                const float weight_tail = transmittance_core * (1.0f - transmittance_tail);
                rgb_pixel += weight_tail * (1.0f / rgba_premultiplied_tail.w) * make_float3(rgba_premultiplied_tail);
                if (!use_median_depth) {
                    depth_pixel += weight_tail * (1.0f / rgba_premultiplied_tail.w) * depth_premultiplied_tail;
                    total_alpha += weight_tail;
                }
            }
            if (!use_median_depth) depth_pixel = (total_alpha > 0.0f) ? depth_pixel / total_alpha : 0.0f;
            // store results

            if constexpr (periphery_mode == PeripheryInterpolationMode::NEAREST) {
                if constexpr (is_lowres_tile) {
                    for (uint x = 0; x < min(config::tile_stride_x, width - pixel_coords.x); x++) {
                        for (uint y = 0; y < min(config::tile_stride_y, height - pixel_coords.y); y++) {
                            const int pixel_idx = width * (pixel_coords.y + y) + (pixel_coords.x + x);
                            if (output_chw) {
                                const int n_pixels = width * height;
                                image[pixel_idx] = __saturatef(rgb_pixel.x);
                                image[n_pixels + pixel_idx] = __saturatef(rgb_pixel.y);
                                image[2 * n_pixels + pixel_idx] = __saturatef(rgb_pixel.z);
                            } else {
                                const int base_idx = 3 * pixel_idx;
                                image[base_idx] = __saturatef(rgb_pixel.x);
                                image[base_idx + 1] = __saturatef(rgb_pixel.y);
                                image[base_idx + 2] = __saturatef(rgb_pixel.z);
                            }
                            depths[pixel_idx] = depth_pixel;
                        }
                    }
                } else {
                    const int pixel_idx = width * pixel_coords.y + pixel_coords.x;
                    if (output_chw) {
                        const int n_pixels = width * height;
                        image[pixel_idx] = __saturatef(rgb_pixel.x);
                        image[n_pixels + pixel_idx] = __saturatef(rgb_pixel.y);
                        image[2 * n_pixels + pixel_idx] = __saturatef(rgb_pixel.z);
                    } else {
                        const int base_idx = 3 * pixel_idx;
                        image[base_idx] = __saturatef(rgb_pixel.x);
                        image[base_idx + 1] = __saturatef(rgb_pixel.y);
                        image[base_idx + 2] = __saturatef(rgb_pixel.z);
                    }
                    depths[pixel_idx] = depth_pixel;
                }
            } else {
                const int pixel_idx = width * pixel_coords.y + pixel_coords.x;
                if (output_chw) {
                    const int n_pixels = width * height;
                    image[pixel_idx] = __saturatef(rgb_pixel.x);
                    image[n_pixels + pixel_idx] = __saturatef(rgb_pixel.y);
                    image[2 * n_pixels + pixel_idx] = __saturatef(rgb_pixel.z);
                } else {
                    const int base_idx = 3 * pixel_idx;
                    image[base_idx] = __saturatef(rgb_pixel.x);
                    image[base_idx + 1] = __saturatef(rgb_pixel.y);
                    image[base_idx + 2] = __saturatef(rgb_pixel.z);
                }
            }

            if constexpr (is_blended_tile) {
                // Get the pixel coordinates of the top left pixel of each blending group
                const uint2 top_left_pixel_coords = make_uint2(
                    large_tile_index_2d.x * config::tile_width_large + subtile_index_2d.x * config::tile_width_small + thread_index.x / config::tile_stride_x * config::tile_stride_x,
                    large_tile_index_2d.y * config::tile_height_large + subtile_index_2d.y * config::tile_height_small + thread_index.y / config::tile_stride_y * config::tile_stride_y
                );

                cooperative_groups::coalesced_group coalesced = cooperative_groups::coalesced_threads();
                coalesced.sync();  // Ensure all threads have written their pixel value

                // Calculate average pixel color for blending
                float3 average_rgb_pixel = make_float3(0.0f);
                // TODO: Test pragma unroll (still requires if though to skip out-of-bounds pixels)
                for (uint x = 0; x < min(config::tile_stride_x, width - top_left_pixel_coords.x); x++) {
                    for (uint y = 0; y < min(config::tile_stride_y, height - top_left_pixel_coords.y); y++) {
                        const int pixel_idx = width * (top_left_pixel_coords.y + y) + (top_left_pixel_coords.x + x);
                        if (output_chw) {
                            const int n_pixels = width * height;
                            average_rgb_pixel.x += image[pixel_idx] / config::num_small_tiles_per_large_tile;
                            average_rgb_pixel.y += image[n_pixels + pixel_idx] / config::num_small_tiles_per_large_tile;
                            average_rgb_pixel.z += image[2 * n_pixels + pixel_idx] / config::num_small_tiles_per_large_tile;
                        } else {
                            const int base_idx = 3 * pixel_idx;
                            average_rgb_pixel.x += image[base_idx] / config::num_small_tiles_per_large_tile;
                            average_rgb_pixel.y += image[base_idx + 1] / config::num_small_tiles_per_large_tile;
                            average_rgb_pixel.z += image[base_idx + 2] / config::num_small_tiles_per_large_tile;
                        }
                    }
                }

                coalesced.sync();  // Ensure all threads have computed the average
                const int pixel_idx = width * pixel_coords.y + pixel_coords.x;

                // Determine blending factor
                const float2 dist_from_gaze = c_gaze_position_cuda - make_float2(pixel_coords.x, pixel_coords.y);
                const float blend_factor = clamp((length(dist_from_gaze) - config::blend_radius) / config::blend_width, 0.0f, 1.0f);

                // Write the average color back to the pixels in the tile
                if (output_chw) {
                    const int n_pixels = width * height;
                    image[pixel_idx] = __saturatef(image[pixel_idx] * (1.0f - blend_factor) + average_rgb_pixel.x * blend_factor);
                    image[n_pixels + pixel_idx] = __saturatef(image[n_pixels + pixel_idx] * (1.0f - blend_factor) + average_rgb_pixel.y * blend_factor);
                    image[2 * n_pixels + pixel_idx] = __saturatef(image[2 * n_pixels + pixel_idx] * (1.0f - blend_factor) + average_rgb_pixel.z * blend_factor);
                } else {
                    const int base_idx = 3 * pixel_idx;
                    image[base_idx] = __saturatef(image[base_idx] * (1.0f - blend_factor) + average_rgb_pixel.x * blend_factor);
                    image[base_idx + 1] = __saturatef(image[base_idx + 1] * (1.0f - blend_factor) + average_rgb_pixel.y * blend_factor);
                    image[base_idx + 2] = __saturatef(image[base_idx + 2] * (1.0f - blend_factor) + average_rgb_pixel.z * blend_factor);
                }
            }
        }
    }

    __global__ void __launch_bounds__(config::block_size_blur) blur_cu(
        float* image,
        float* image_blurred,
        const uint width,
        const uint height,
        const uint grid_width,
        const uint2 gaze_position_tiles,
        const bool output_chw
    ) {
        constexpr float gaussian_kernel_factors[3][3] = {
            {1.0f, 2.0f, 1.0f},
            {2.0f, 4.0f, 2.0f},
            {1.0f, 2.0f, 1.0f}
        };
        constexpr float gaussian_factor = 1.0f / 16.0f;

        const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        const dim3 group_index = block.group_index();
        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        const uint large_tile_index = group_index.y / config::tile_stride_y * grid_width + group_index.x / config::tile_stride_x;
        const bool is_lowres_tile = !is_in_fovea<config::foveation_radius_tiles>(large_tile_index, grid_width, gaze_position_tiles);

        const uint2 pixel_coords = make_uint2(
            group_index.x * config::tile_width_small + thread_index.x,
            group_index.y * config::tile_height_small + thread_index.y
        );
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        if (!inside) return;
        const int pixel_idx = width * pixel_coords.y + pixel_coords.x;

        if (is_lowres_tile) {
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
        } else {
            if (output_chw) {
                const int n_pixels = width * height;
                image_blurred[pixel_idx] = image[pixel_idx];
                image_blurred[n_pixels + pixel_idx] = image[n_pixels + pixel_idx];
                image_blurred[2 * n_pixels + pixel_idx] = image[2 * n_pixels + pixel_idx];
            } else {
                const int base_idx = 3 * pixel_idx;
                image_blurred[base_idx] = image[base_idx];
                image_blurred[base_idx + 1] = image[base_idx + 1];
                image_blurred[base_idx + 2] = image[base_idx + 2];
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

    __global__ void __launch_bounds__(config::gaze_visualization_size) visualize_gaze(
        float* image,
        const uint width,
        const uint height,
        const bool output_chw
    ) {
        const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        const dim3 thread_index = block.thread_index();
        const int x_off = thread_index.x - config::gaze_visualization_width / 2;
        const int y_off = thread_index.y - config::gaze_visualization_width / 2;

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
