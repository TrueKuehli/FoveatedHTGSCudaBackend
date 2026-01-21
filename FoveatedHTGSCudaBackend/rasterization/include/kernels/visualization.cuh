#include "config.h"
#include "helper_math.h"
#include "utils/kernel_utils.cuh"
#include <cooperative_groups.h>


namespace htgs_foveated::rasterization::kernels::visualization {

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
