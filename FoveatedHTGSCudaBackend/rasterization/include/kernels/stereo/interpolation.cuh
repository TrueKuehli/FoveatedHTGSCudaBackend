#pragma once

#include "config.h"
#include "helper_math.h"
#include "utils/stereo/kernel_utils.cuh"
#include <cooperative_groups.h>


namespace htgs_foveated::rasterization::kernels::stereo::interpolation {

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

        // The following assumes pixel stride of 2
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


    template<bool second_camera>
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
        const float2 dist_from_gaze = c_gaze_position_cuda[second_camera] - make_float2(pixel_x, pixel_y);
        const float blend_factor = clamp((length(dist_from_gaze) - config::blend_radius) / config::blend_width, 0.0f, 1.0f);

        // The following assumes pixel stride of 2
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
        float* dst,
        float* src,
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
        const dim3 subtile_index_2d(subtile_index % config::tile_stride_x, subtile_index / config::tile_stride_x, 0);
        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        const uint pixel_offset_x = block.group_index().y % config::tile_stride_x;
        const uint pixel_offset_y = block.group_index().y / config::tile_stride_x;

        const uint2 pixel_coords = make_uint2(
            large_tile_index_2d.x * config::tile_width_large + subtile_index_2d.x * config::tile_width_small + thread_index.x,
            large_tile_index_2d.y * config::tile_height_large + subtile_index_2d.y * config::tile_height_small + thread_index.y
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
}