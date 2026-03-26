#include "config.h"
#include "helper_math.h"
#include "utils/monocular/kernel_utils.cuh"
#include "visualization_config.h"
#include <cooperative_groups.h>


namespace htgs_foveated::visualization::kernels {

    __global__ void __launch_bounds__(visualization::config::gaze_visualization_size) visualize_gaze(
        float* image,
        const uint width,
        const uint height,
        const bool output_chw
    ) {
        const dim3 group_index = cooperative_groups::this_thread_block().group_index();
        const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        const dim3 thread_index = block.thread_index();
        const int x_off = group_index.x * rasterization::config::tile_width_small + thread_index.x - visualization::config::gaze_visualization_width / 2;
        const int y_off = group_index.y * rasterization::config::tile_width_small + thread_index.y - visualization::config::gaze_visualization_width / 2;

        if constexpr (visualization::config::gaze_visualization_circular) {
            if (x_off * x_off + y_off * y_off > (visualization::config::gaze_visualization_width / 2) * (visualization::config::gaze_visualization_width / 2)) return;
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

    __global__ void __launch_bounds__(visualization::config::num_border_pixels_large) visualize_tile_boundaries_cu(
        float* image,
        const float3 color,
        const uint* tile_index_map,
        const uint tile_offset,
        const uint width,
        const uint height,
        const uint tile_width,
        const uint tile_height,
        const uint grid_width,
        const bool output_chw)
    {
        const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        const uint group_index = block.group_index().x + tile_offset;
        const uint true_group_index = tile_index_map[group_index];
        const uint large_tile_index = true_group_index / rasterization::config::num_small_tiles_per_large_tile;
        const uint subtile_index = true_group_index % rasterization::config::num_small_tiles_per_large_tile;
        const dim3 large_tile_index_2d(large_tile_index % grid_width, large_tile_index / grid_width, 0);
        const dim3 subtile_index_2d(subtile_index % rasterization::config::tile_stride_x, subtile_index / rasterization::config::tile_stride_x, 0);
        const int thread_index = block.thread_index().x;

        // Map linear thread index to tile border pixel coordinates
        int tile_pixel_x;
        int tile_pixel_y;
        if (thread_index < tile_width) {
            // top border
            tile_pixel_x = thread_index;
            tile_pixel_y = 0;
        } else if (thread_index < 2 * tile_width) {
            // bottom border
            tile_pixel_x = thread_index - tile_width;
            tile_pixel_y = tile_height - 1;
        } else if (thread_index < 2 * tile_width + tile_height - 2) {
            // left border
            tile_pixel_x = 0;
            tile_pixel_y = thread_index - 2 * tile_width + 1;
        } else {
            tile_pixel_x = tile_width - 1;
            tile_pixel_y = thread_index - (2 * tile_width + tile_height - 2) + 1;
        }
        const uint2 pixel_coords = make_uint2(
            large_tile_index_2d.x * rasterization::config::tile_width_large + subtile_index_2d.x * rasterization::config::tile_width_small + tile_pixel_x,
            large_tile_index_2d.y * rasterization::config::tile_height_large + subtile_index_2d.y * rasterization::config::tile_height_small + tile_pixel_y
        );
        if (pixel_coords.x >= width || pixel_coords.y >= height) return;
        const int pixel_idx = width * pixel_coords.y + pixel_coords.x;

        // Write color
        if (output_chw) {
            const int n_pixels = width * height;
            image[pixel_idx] = color.x;
            image[n_pixels + pixel_idx] = color.y;
            image[2 * n_pixels + pixel_idx] = color.z;
        } else {
            const int base_idx = 3 * pixel_idx;
            image[base_idx] = color.x;
            image[base_idx + 1] = color.y;
            image[base_idx + 2] = color.z;
        }
    }
}
