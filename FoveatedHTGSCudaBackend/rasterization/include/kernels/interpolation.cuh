#pragma once

#include "config.h"
#include "helper_math.h"
#include "utils/kernel_utils.cuh"
#include <cooperative_groups.h>


namespace htgs_foveated::rasterization::kernels::interpolation {

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
        const int sample_idx = width * sample_y + sample_x;

        if (output_chw) {
            const int n_pixels = width * height;
            return make_float3(
                image[sample_idx],
                image[n_pixels + sample_idx],
                image[2 * n_pixels + sample_idx]
            );
        } else {
            const int base_idx = 3 * sample_idx;
            return make_float3(
                image[base_idx],
                image[base_idx + 1],
                image[base_idx + 2]
            );
        }
    }


    __global__ void __launch_bounds__(config::block_size_blur) bilinear_interpolation(
        float* image,
        const uint32_t* visibility_mask,
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
        const uint thread_rank = block.thread_rank();
        int base_pixel_x = large_tile_index_2d.x * config::tile_width_large;
        int base_pixel_y = large_tile_index_2d.y * config::tile_height_large;

        // We get one sample below/right of the tile boundary; with the number of samples in the tile
        //   boundary being equal to the number of pixels in a full-resolution tile
        constexpr const int num_sample_points_x = config::tile_width_small + 1;
        constexpr const int num_sample_points_y = config::tile_height_small + 1;
        constexpr const int num_sample_points = num_sample_points_x * num_sample_points_y;
        constexpr const int n_iters_loading = (num_sample_points + config::block_size_blur - 1) / config::block_size_blur; // ceil division
        __shared__ float3 sample_points[num_sample_points_y][num_sample_points_x];

        // Collaborative loading of sample points
        for (int i = 0; i < n_iters_loading; i++) {
            block.sync();
            const int current_fetch_idx = thread_rank + i * config::block_size_blur;
            if (current_fetch_idx < num_sample_points) {
                int sample_x = current_fetch_idx % num_sample_points_x;
                int sample_y = current_fetch_idx / num_sample_points_x;
                int pixel_x = base_pixel_x + sample_x * config::tile_stride_x;
                int pixel_y = base_pixel_y + sample_y * config::tile_stride_y;

                const int sample_tile_idx = pixel_y / config::tile_height_large * grid_width + pixel_x / config::tile_width_large;
                const int mask_byte_idx = sample_tile_idx / 32;
                const int mask_bit_idx = sample_tile_idx % 32;
                if ((visibility_mask[mask_byte_idx] & (1 << mask_bit_idx)) == 0) {
                    // Clamp to nearest valid pixel within the tile boundary if outside the image boundary; this is to avoid artifacts from sampling black pixels outside the image
                    int clamped_pixel_x = clamp(pixel_x,
                            static_cast<int>(large_tile_index_2d.x * config::tile_width_large),
                            static_cast<int>(large_tile_index_2d.x * config::tile_width_large + config::tile_width_large - config::tile_stride_x));
                    int clamped_pixel_y = clamp(pixel_y,
                            static_cast<int>(large_tile_index_2d.y * config::tile_height_large),
                            static_cast<int>(large_tile_index_2d.y * config::tile_height_large + config::tile_height_large - config::tile_stride_y));
                    sample_points[sample_y][sample_x] = sample_rgb(image, clamped_pixel_x, clamped_pixel_y, width, height, output_chw);
                } else {
                    sample_points[sample_y][sample_x] = sample_rgb(image, pixel_x, pixel_y, width, height, output_chw);
                }
            }
            block.sync();
        }

        // Sample idx of top left pixel
        int base_sample_x = thread_index.x / config::tile_stride_x;
        int base_sample_y = thread_index.y / config::tile_stride_y;

        int pixel_x = large_tile_index_2d.x * config::tile_width_large + thread_index.x;
        int pixel_y = large_tile_index_2d.y * config::tile_height_large + thread_index.y;
        const bool inside = pixel_x < width && pixel_y < height;
        if (!inside) return;

        float3 interpolated_rgb = make_float3(0.0f);
        uint x_mod = thread_index.x % config::tile_stride_x;
        uint y_mod = thread_index.y % config::tile_stride_y;
        float x_frac = float(x_mod) / float(config::tile_stride_x);
        float y_frac = float(y_mod) / float(config::tile_stride_y);

        if ((thread_index.x % config::tile_stride_x) == 0 && (thread_index.y % config::tile_stride_y) == 0) {
            // Sample position, value already set in image
            return;
        } else if ((thread_index.x % config::tile_stride_x) == 0) {  // x is directly on the sample
            interpolated_rgb += sample_points[base_sample_y    ][base_sample_x] * (1.0f - y_frac);
            interpolated_rgb += sample_points[base_sample_y + 1][base_sample_x] * y_frac;
        } else if ((thread_index.y % config::tile_stride_y) == 0) {  // y is directly on the sample
            interpolated_rgb += sample_points[base_sample_y][base_sample_x    ] * (1.0f - x_frac);
            interpolated_rgb += sample_points[base_sample_y][base_sample_x + 1] * x_frac;
        } else {
            interpolated_rgb += sample_points[base_sample_y    ][base_sample_x    ] * (1.0f - x_frac) * (1.0f - y_frac);
            interpolated_rgb += sample_points[base_sample_y    ][base_sample_x + 1] * x_frac          * (1.0f - y_frac);
            interpolated_rgb += sample_points[base_sample_y + 1][base_sample_x    ] * (1.0f - x_frac) * y_frac;
            interpolated_rgb += sample_points[base_sample_y + 1][base_sample_x + 1] * x_frac          * y_frac;
        }

        const int pixel_idx = width * pixel_y + pixel_x;
        if (output_chw) {
            const int n_pixels = width * height;
            image[pixel_idx] = __saturatef(interpolated_rgb.x);
            image[n_pixels + pixel_idx] = __saturatef(interpolated_rgb.y);
            image[2 * n_pixels + pixel_idx] = __saturatef(interpolated_rgb.z);
        } else {
            const int base_idx = 3 * pixel_idx;
            image[base_idx] = __saturatef(interpolated_rgb.x);
            image[base_idx + 1] = __saturatef(interpolated_rgb.y);
            image[base_idx + 2] = __saturatef(interpolated_rgb.z);
        }
    }


    template<uint8_t cam_idx>
    __global__ void __launch_bounds__(config::block_size_blur_blended) bilinear_interpolation_blended(
        float* image,
        const uint32_t* visibility_mask,
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
        const uint subtile_index = true_group_index % config::num_small_tiles_per_large_tile;
        const dim3 large_tile_index_2d(large_tile_index % grid_width, large_tile_index / grid_width, 0);
        const dim3 subtile_index_2d(subtile_index % config::tile_stride_x, subtile_index / config::tile_stride_y, 0);

        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        int base_pixel_x = large_tile_index_2d.x * config::tile_width_large + subtile_index_2d.x * config::tile_width_small;
        int base_pixel_y = large_tile_index_2d.y * config::tile_height_large + subtile_index_2d.y * config::tile_height_small;

        // We get one sample below/right of the tile boundary
        constexpr const int num_sample_points_x = config::tile_width_small / config::tile_stride_x + 1;
        constexpr const int num_sample_points_y = config::tile_height_small / config::tile_stride_y + 1;
        constexpr const int num_sample_points = num_sample_points_x * num_sample_points_y;
        constexpr const int n_iters_loading = (num_sample_points + config::block_size_blur_blended - 1) / config::block_size_blur_blended; // ceil division
        __shared__ float3 sample_points[num_sample_points_y][num_sample_points_x];

        // Collaborative loading of sample points
        for (int i = 0; i < n_iters_loading; i++) {
            block.sync();
            const int current_fetch_idx = thread_rank + i * config::block_size_blur_blended;
            if (current_fetch_idx < num_sample_points) {
                int sample_x = current_fetch_idx % num_sample_points_x;
                int sample_y = current_fetch_idx / num_sample_points_x;
                int pixel_x = base_pixel_x + sample_x * config::tile_stride_x;
                int pixel_y = base_pixel_y + sample_y * config::tile_stride_y;

                const int sample_tile_idx = pixel_y / config::tile_height_large * grid_width + pixel_x / config::tile_width_large;
                const int mask_byte_idx = sample_tile_idx / 32;
                const int mask_bit_idx = sample_tile_idx % 32;
                if ((visibility_mask[mask_byte_idx] & (1 << mask_bit_idx)) == 0) {
                    // Clamp to nearest valid pixel within the tile boundary if outside the image boundary; this is to avoid artifacts from sampling black pixels outside the image
                    int clamped_pixel_x = clamp(pixel_x,
                            static_cast<int>(large_tile_index_2d.x * config::tile_width_large),
                            static_cast<int>(large_tile_index_2d.x * config::tile_width_large + config::tile_width_large - config::tile_stride_x));
                    int clamped_pixel_y = clamp(pixel_y,
                            static_cast<int>(large_tile_index_2d.y * config::tile_height_large),
                            static_cast<int>(large_tile_index_2d.y * config::tile_height_large + config::tile_height_large - config::tile_stride_y));
                    sample_points[sample_y][sample_x] = sample_rgb(image, clamped_pixel_x, clamped_pixel_y, width, height, output_chw);
                } else {
                    sample_points[sample_y][sample_x] = sample_rgb(image, pixel_x, pixel_y, width, height, output_chw);
                }
            }
            block.sync();
        }

        // Sample idx of top left pixel
        int base_sample_x = thread_index.x / config::tile_stride_x;
        int base_sample_y = thread_index.y / config::tile_stride_y;

        int pixel_x = large_tile_index_2d.x * config::tile_width_large + subtile_index_2d.x * config::tile_width_small + thread_index.x;
        int pixel_y = large_tile_index_2d.y * config::tile_height_large + subtile_index_2d.y * config::tile_height_small + thread_index.y;
        const bool inside = pixel_x < width && pixel_y < height;
        if (!inside) return;

        float3 interpolated_rgb = make_float3(0.0f);
        uint x_mod = thread_index.x % config::tile_stride_x;
        uint y_mod = thread_index.y % config::tile_stride_y;
        float x_frac = float(x_mod) / float(config::tile_stride_x);
        float y_frac = float(y_mod) / float(config::tile_stride_y);

        if ((thread_index.x % config::tile_stride_x) == 0 && (thread_index.y % config::tile_stride_y) == 0) {
            // Sample position, value already set in image
            return;
        } else if ((thread_index.x % config::tile_stride_x) == 0) {  // x is directly on the sample
            interpolated_rgb += sample_points[base_sample_y    ][base_sample_x] * (1.0f - y_frac);
            interpolated_rgb += sample_points[base_sample_y + 1][base_sample_x] * y_frac;
        } else if ((thread_index.y % config::tile_stride_y) == 0) {  // y is directly on the sample
            interpolated_rgb += sample_points[base_sample_y][base_sample_x    ] * (1.0f - x_frac);
            interpolated_rgb += sample_points[base_sample_y][base_sample_x + 1] * x_frac;
        } else {
            interpolated_rgb += sample_points[base_sample_y    ][base_sample_x    ] * (1.0f - x_frac) * (1.0f - y_frac);
            interpolated_rgb += sample_points[base_sample_y    ][base_sample_x + 1] * x_frac          * (1.0f - y_frac);
            interpolated_rgb += sample_points[base_sample_y + 1][base_sample_x    ] * (1.0f - x_frac) * y_frac;
            interpolated_rgb += sample_points[base_sample_y + 1][base_sample_x + 1] * x_frac          * y_frac;
        }

        // Determine blending factor
        const float2 dist_from_gaze = c_gaze_position_cuda[cam_idx] - make_float2(pixel_x, pixel_y);
        const float blend_factor = clamp(
                (length(dist_from_gaze) - config::blend_radius - config::tile_width_large)
                / (config::blend_width - config::tile_width_large),
                0.0f, 1.0f
        );

        float3 center_rgb = sample_rgb(image, pixel_x, pixel_y, width, height, output_chw);
        const int pixel_idx = width * pixel_y + pixel_x;
        if (blend_factor > 0.0f) {
            if (output_chw) {
                const int n_pixels = width * height;
                image[pixel_idx] = __saturatef(interpolated_rgb.x * blend_factor + center_rgb.x * (1.0f - blend_factor));
                image[n_pixels + pixel_idx] = __saturatef(interpolated_rgb.y * blend_factor + center_rgb.y * (1.0f - blend_factor));
                image[2 * n_pixels + pixel_idx] = __saturatef(interpolated_rgb.z * blend_factor + center_rgb.z * (1.0f - blend_factor));
            } else {
                const int base_idx = 3 * pixel_idx;
                image[base_idx] = __saturatef(interpolated_rgb.x * blend_factor + center_rgb.x * (1.0f - blend_factor));
                image[base_idx + 1] = __saturatef(interpolated_rgb.y * blend_factor + center_rgb.y * (1.0f - blend_factor));
                image[base_idx + 2] = __saturatef(interpolated_rgb.z * blend_factor + center_rgb.z * (1.0f - blend_factor));
            }
        }
    }


    __global__ void __launch_bounds__(config::block_size_blur) interpolate_and_blur(
        float* image_blurred,
        const float* image,
        const uint32_t* visibility_mask,
        const uint* tile_index_map,
        const uint tile_offset,
        const uint width,
        const uint height,
        const uint grid_width,
        const bool output_chw
    ) {
        // Fused bilinear interpolation + gaussian blur kernel
        const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        const uint group_index = block.group_index().x + tile_offset;
        const uint true_group_index = tile_index_map[group_index];
        const uint large_tile_index = true_group_index / config::num_small_tiles_per_large_tile;
        const dim3 large_tile_index_2d(large_tile_index % grid_width, large_tile_index / grid_width, 0);

        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        int base_pixel_x = large_tile_index_2d.x * config::tile_width_large - config::tile_stride_x;
        int base_pixel_y = large_tile_index_2d.y * config::tile_height_large - config::tile_stride_y;

        // We get one sample above/below/left/right of the tile boundary; with the number of samples in the tile
        //   boundary being equal to the number of pixels in a full-resolution tile
        constexpr const int num_sample_points_x = config::tile_width_small + 2;
        constexpr const int num_sample_points_y = config::tile_height_small + 2;
        constexpr const int num_sample_points = num_sample_points_x * num_sample_points_y;
        constexpr const int n_iters_loading = (num_sample_points + config::block_size_blur - 1) / config::block_size_blur; // ceil division
        __shared__ float3 sample_points[num_sample_points_y][num_sample_points_x];

        // Collaborative loading of sample points
        for (int i = 0; i < n_iters_loading; i++) {
            block.sync();
            const int current_fetch_idx = thread_rank + i * config::block_size_blur;
            if (current_fetch_idx < num_sample_points) {
                int sample_x = current_fetch_idx % num_sample_points_x;
                int sample_y = current_fetch_idx / num_sample_points_x;
                int pixel_x = base_pixel_x + sample_x * config::tile_stride_x;
                int pixel_y = base_pixel_y + sample_y * config::tile_stride_y;

                const int sample_tile_idx = pixel_y / config::tile_height_large * grid_width + pixel_x / config::tile_width_large;
                const int mask_byte_idx = sample_tile_idx / 32;
                const int mask_bit_idx = sample_tile_idx % 32;
                if ((visibility_mask[mask_byte_idx] & (1 << mask_bit_idx)) == 0) {
                    // Clamp to nearest valid pixel within the tile boundary if outside the image boundary; this is to avoid artifacts from sampling black pixels outside the image
                    int clamped_pixel_x = clamp(pixel_x,
                            static_cast<int>(large_tile_index_2d.x * config::tile_width_large),
                            static_cast<int>(large_tile_index_2d.x * config::tile_width_large + config::tile_width_large - config::tile_stride_x));
                    int clamped_pixel_y = clamp(pixel_y,
                            static_cast<int>(large_tile_index_2d.y * config::tile_height_large),
                            static_cast<int>(large_tile_index_2d.y * config::tile_height_large + config::tile_height_large - config::tile_stride_y));
                    sample_points[sample_y][sample_x] = sample_rgb(image, clamped_pixel_x, clamped_pixel_y, width, height, output_chw);
                } else {
                    sample_points[sample_y][sample_x] = sample_rgb(image, pixel_x, pixel_y, width, height, output_chw);
                }
            }
            block.sync();
        }

        // Sample idx of top left pixel
        int base_sample_x = thread_index.x / 2 + 1;
        int base_sample_y = thread_index.y / 2 + 1;

        int pixel_x = large_tile_index_2d.x * config::tile_width_large + thread_index.x;
        int pixel_y = large_tile_index_2d.y * config::tile_height_large + thread_index.y;
        const bool inside = pixel_x < width && pixel_y < height;
        if (!inside) return;

        // The following assumes pixel stride of 2
        float3 blurred_rgb = make_float3(0.0f);
        float blur_factor;
        if ((thread_index.x & 0b1) == 0 && (thread_index.y & 0b1) == 0) {
            // Top left pixel, factors:
            //   [1.0   0.0   6.0  0.0  1.0]
            //   [0.0   0.0   0.0  0.0  0.0]
            //   [6.0   0.0  36.0  0.0  6.0]
            //   [0.0   0.0   0.0  0.0  0.0]
            //   [1.0   0.0   6.0  0.0  1.0]
            blur_factor = 1.0f / 64.0f;
            blurred_rgb += sample_points[base_sample_y - 1][base_sample_x - 1];
            blurred_rgb += sample_points[base_sample_y - 1][base_sample_x    ] *  6.0f;
            blurred_rgb += sample_points[base_sample_y - 1][base_sample_x + 1];
            blurred_rgb += sample_points[base_sample_y    ][base_sample_x - 1] *  6.0f;
            blurred_rgb += sample_points[base_sample_y    ][base_sample_x    ] * 36.0f;
            blurred_rgb += sample_points[base_sample_y    ][base_sample_x + 1] *  6.0f;
            blurred_rgb += sample_points[base_sample_y + 1][base_sample_x - 1];
            blurred_rgb += sample_points[base_sample_y + 1][base_sample_x    ] *  6.0f;
            blurred_rgb += sample_points[base_sample_y + 1][base_sample_x + 1];
        } else if ((thread_index.x & 0b1) == 0 && (thread_index.y & 0b1) == 1) {
            // Bottom left pixel, factors:
            //   [1.0  0.0  6.0  0.0  1.0]
            //   [0.0  0.0  0.0  0.0  0.0]
            //   [1.0  0.0  6.0  0.0  1.0]
            blur_factor = 1.0f / 16.0f;
            blurred_rgb += sample_points[base_sample_y    ][base_sample_x - 1];
            blurred_rgb += sample_points[base_sample_y    ][base_sample_x    ] * 6.0f;
            blurred_rgb += sample_points[base_sample_y    ][base_sample_x + 1];
            blurred_rgb += sample_points[base_sample_y + 1][base_sample_x - 1];
            blurred_rgb += sample_points[base_sample_y + 1][base_sample_x    ] * 6.0f;
            blurred_rgb += sample_points[base_sample_y + 1][base_sample_x + 1];
        } else if ((thread_index.x & 0b1) == 1 && (thread_index.y & 0b1) == 0) {
            // Top right pixel, factors:
            //   [1.0  0.0  1.0]
            //   [0.0  0.0  0.0]
            //   [6.0  0.0  6.0]
            //   [0.0  0.0  0.0]
            //   [1.0  0.0  1.0]
            blur_factor = 1.0f / 16.0f;
            blurred_rgb += sample_points[base_sample_y - 1][base_sample_x    ];
            blurred_rgb += sample_points[base_sample_y - 1][base_sample_x + 1];
            blurred_rgb += sample_points[base_sample_y    ][base_sample_x    ] * 6.0f;
            blurred_rgb += sample_points[base_sample_y    ][base_sample_x + 1] * 6.0f;
            blurred_rgb += sample_points[base_sample_y + 1][base_sample_x    ];
            blurred_rgb += sample_points[base_sample_y + 1][base_sample_x + 1];
        } else { // if ((thread_index.x & 0b1) == 1 && (thread_index.y & 0b1) == 1) {
            // Bottom right pixel, factors:
            //   [1.0  0.0  1.0]
            //   [0.0  0.0  0.0]
            //   [1.0  0.0  1.0]
            blur_factor = 1.0f / 4.0f;
            blurred_rgb += sample_points[base_sample_y    ][base_sample_x    ];
            blurred_rgb += sample_points[base_sample_y    ][base_sample_x + 1];
            blurred_rgb += sample_points[base_sample_y + 1][base_sample_x    ];
            blurred_rgb += sample_points[base_sample_y + 1][base_sample_x + 1];
        }

        const int pixel_idx = width * pixel_y + pixel_x;
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


    template<uint8_t cam_idx>
    __global__ void __launch_bounds__(config::block_size_blur_blended) interpolate_and_blur_blended(
        float* image_blurred,
        const float* image,
        const uint32_t* visibility_mask,
        const uint* tile_index_map,
        const uint tile_offset,
        const uint width,
        const uint height,
        const uint grid_width,
        const bool output_chw
    ) {
        // Fused bilinear interpolation + gaussian blur kernel; with output blended with existing pixels (for blended tiles)
        const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        const uint group_index = block.group_index().x + tile_offset;
        const uint true_group_index = tile_index_map[group_index];
        const uint large_tile_index = true_group_index / config::num_small_tiles_per_large_tile;
        const uint subtile_index = true_group_index % config::num_small_tiles_per_large_tile;
        const dim3 large_tile_index_2d(large_tile_index % grid_width, large_tile_index / grid_width, 0);
        const dim3 subtile_index_2d(subtile_index % config::tile_stride_x, subtile_index / config::tile_stride_y, 0);

        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        int base_pixel_x = large_tile_index_2d.x * config::tile_width_large + subtile_index_2d.x * config::tile_width_small - config::tile_stride_x;
        int base_pixel_y = large_tile_index_2d.y * config::tile_height_large + subtile_index_2d.y * config::tile_height_small - config::tile_stride_y;

        // We get one sample above/below/left/right of the tile boundary
        constexpr const int num_sample_points_x = config::tile_width_small / config::tile_stride_x + 2;
        constexpr const int num_sample_points_y = config::tile_height_small / config::tile_stride_y + 2;
        constexpr const int num_sample_points = num_sample_points_x * num_sample_points_y;
        constexpr const int n_iters_loading = (num_sample_points + config::block_size_blur_blended - 1) / config::block_size_blur_blended; // ceil division
        __shared__ float3 sample_points[num_sample_points_y][num_sample_points_x];

        // Collaborative loading of sample points
        for (int i = 0; i < n_iters_loading; i++) {
            block.sync();
            const int current_fetch_idx = thread_rank + i * config::block_size_blur_blended;
            if (current_fetch_idx < num_sample_points) {
                int sample_x = current_fetch_idx % num_sample_points_x;
                int sample_y = current_fetch_idx / num_sample_points_x;
                int pixel_x = base_pixel_x + sample_x * config::tile_stride_x;
                int pixel_y = base_pixel_y + sample_y * config::tile_stride_y;

                const int sample_tile_idx = pixel_y / config::tile_height_large * grid_width + pixel_x / config::tile_width_large;
                const int mask_byte_idx = sample_tile_idx / 32;
                const int mask_bit_idx = sample_tile_idx % 32;
                if ((visibility_mask[mask_byte_idx] & (1 << mask_bit_idx)) == 0) {
                    // Clamp to nearest valid pixel within the tile boundary if outside the image boundary; this is to avoid artifacts from sampling black pixels outside the image
                    int clamped_pixel_x = clamp(pixel_x,
                            static_cast<int>(large_tile_index_2d.x * config::tile_width_large),
                            static_cast<int>(large_tile_index_2d.x * config::tile_width_large + config::tile_width_large - config::tile_stride_x));
                    int clamped_pixel_y = clamp(pixel_y,
                            static_cast<int>(large_tile_index_2d.y * config::tile_height_large),
                            static_cast<int>(large_tile_index_2d.y * config::tile_height_large + config::tile_height_large - config::tile_stride_y));
                    sample_points[sample_y][sample_x] = sample_rgb(image, clamped_pixel_x, clamped_pixel_y, width, height, output_chw);
                } else {
                    sample_points[sample_y][sample_x] = sample_rgb(image, pixel_x, pixel_y, width, height, output_chw);
                }
            }
            block.sync();
        }

        // Sample idx of top left pixel
        int base_sample_x = thread_index.x / 2 + 1;
        int base_sample_y = thread_index.y / 2 + 1;

        int pixel_x = large_tile_index_2d.x * config::tile_width_large + subtile_index_2d.x * config::tile_width_small + thread_index.x;
        int pixel_y = large_tile_index_2d.y * config::tile_height_large + subtile_index_2d.y * config::tile_height_small + thread_index.y;
        const bool inside = pixel_x < width && pixel_y < height;
        if (!inside) return;

        // The following assumes pixel stride of 2
        float3 blurred_rgb = make_float3(0.0f);
        float blur_factor;
        if ((thread_index.x & 0b1) == 0 && (thread_index.y & 0b1) == 0) {
            // Top left pixel, factors:
            //   [1.0   0.0   6.0  0.0  1.0]
            //   [0.0   0.0   0.0  0.0  0.0]
            //   [6.0   0.0  36.0  0.0  6.0]
            //   [0.0   0.0   0.0  0.0  0.0]
            //   [1.0   0.0   6.0  0.0  1.0]
            blur_factor = 1.0f / 64.0f;
            blurred_rgb += sample_points[base_sample_y - 1][base_sample_x - 1];
            blurred_rgb += sample_points[base_sample_y - 1][base_sample_x    ] *  6.0f;
            blurred_rgb += sample_points[base_sample_y - 1][base_sample_x + 1];
            blurred_rgb += sample_points[base_sample_y    ][base_sample_x - 1] *  6.0f;
            blurred_rgb += sample_points[base_sample_y    ][base_sample_x    ] * 36.0f;
            blurred_rgb += sample_points[base_sample_y    ][base_sample_x + 1] *  6.0f;
            blurred_rgb += sample_points[base_sample_y + 1][base_sample_x - 1];
            blurred_rgb += sample_points[base_sample_y + 1][base_sample_x    ] *  6.0f;
            blurred_rgb += sample_points[base_sample_y + 1][base_sample_x + 1];
        } else if ((thread_index.x & 0b1) == 0 && (thread_index.y & 0b1) == 1) {
            // Bottom left pixel, factors:
            //   [1.0  0.0  6.0  0.0  1.0]
            //   [0.0  0.0  0.0  0.0  0.0]
            //   [1.0  0.0  6.0  0.0  1.0]
            blur_factor = 1.0f / 16.0f;
            blurred_rgb += sample_points[base_sample_y    ][base_sample_x - 1];
            blurred_rgb += sample_points[base_sample_y    ][base_sample_x    ] * 6.0f;
            blurred_rgb += sample_points[base_sample_y    ][base_sample_x + 1];
            blurred_rgb += sample_points[base_sample_y + 1][base_sample_x - 1];
            blurred_rgb += sample_points[base_sample_y + 1][base_sample_x    ] * 6.0f;
            blurred_rgb += sample_points[base_sample_y + 1][base_sample_x + 1];
        } else if ((thread_index.x & 0b1) == 1 && (thread_index.y & 0b1) == 0) {
            // Top right pixel, factors:
            //   [1.0  0.0  1.0]
            //   [0.0  0.0  0.0]
            //   [6.0  0.0  6.0]
            //   [0.0  0.0  0.0]
            //   [1.0  0.0  1.0]
            blur_factor = 1.0f / 16.0f;
            blurred_rgb += sample_points[base_sample_y - 1][base_sample_x    ];
            blurred_rgb += sample_points[base_sample_y - 1][base_sample_x + 1];
            blurred_rgb += sample_points[base_sample_y    ][base_sample_x    ] * 6.0f;
            blurred_rgb += sample_points[base_sample_y    ][base_sample_x + 1] * 6.0f;
            blurred_rgb += sample_points[base_sample_y + 1][base_sample_x    ];
            blurred_rgb += sample_points[base_sample_y + 1][base_sample_x + 1];
        } else { // if ((thread_index.x & 0b1) == 1 && (thread_index.y & 0b1) == 1) {
            // Bottom right pixel, factors:
            //   [1.0  0.0  1.0]
            //   [0.0  0.0  0.0]
            //   [1.0  0.0  1.0]
            blur_factor = 1.0f / 4.0f;
            blurred_rgb += sample_points[base_sample_y    ][base_sample_x    ];
            blurred_rgb += sample_points[base_sample_y    ][base_sample_x + 1];
            blurred_rgb += sample_points[base_sample_y + 1][base_sample_x    ];
            blurred_rgb += sample_points[base_sample_y + 1][base_sample_x + 1];
        }

        // Determine blending factor
        const float2 dist_from_gaze = c_gaze_position_cuda[cam_idx] - make_float2(pixel_x, pixel_y);
        const float blend_factor = clamp(
                (length(dist_from_gaze) - config::blend_radius - config::tile_width_large)
                / (config::blend_width - config::tile_width_large),
                0.0f, 1.0f
        );

        float3 center_rgb = sample_rgb(image, pixel_x, pixel_y, width, height, output_chw);
        const int pixel_idx = width * pixel_y + pixel_x;
        if (blend_factor > 0.0f) {
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
        } else {
            if (output_chw) {
                const int n_pixels = width * height;
                image_blurred[pixel_idx] = center_rgb.x;
                image_blurred[n_pixels + pixel_idx] = center_rgb.y;
                image_blurred[2 * n_pixels + pixel_idx] = center_rgb.z;
            } else {
                const int base_idx = 3 * pixel_idx;
                image_blurred[base_idx] = center_rgb.x;
                image_blurred[base_idx + 1] = center_rgb.y;
                image_blurred[base_idx + 2] = center_rgb.z;
            }
        }
    }
}
