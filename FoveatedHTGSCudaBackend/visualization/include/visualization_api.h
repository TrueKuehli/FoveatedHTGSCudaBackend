#pragma once

#include <torch/extension.h>


namespace htgs_foveated::visualization {

    torch::Tensor visualize_gaze_wrapper(
        const torch::Tensor& image,
        const torch::Tensor& gaze_position,
        const torch::Tensor& gaze_color,
        const uint8_t visualization_type,
        const int visualization_size,
        const bool to_chw);

    torch::Tensor visualize_tile_boundaries_wrapper(
        const torch::Tensor& image,
        const torch::Tensor& render_mask,
        const torch::Tensor& gaze_position,
        const bool to_chw);

}
