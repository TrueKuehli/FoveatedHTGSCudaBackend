#pragma once

#include <torch/extension.h>
#include <tuple>

namespace htgs::rasterization {
    std::tuple<torch::Tensor, torch::Tensor>
    inference_wrapper(
        const torch::Tensor& positions,
        const torch::Tensor& scales,
        const torch::Tensor& rotations,
        const torch::Tensor& opacities,
        const torch::Tensor& sh_0,
        const torch::Tensor& sh_rest,
        const torch::Tensor& M,
        const torch::Tensor& VPM,
        const torch::Tensor& cam_position,
        const torch::Tensor& gaze_position,
        const torch::Tensor& render_mask,
        const torch::Tensor& render_mask_area_table,
        const torch::Tensor& fovea_mask_area_table,
        const int periphery_interpolation_mode,
        const int K,
        const int active_sh_bases,
        const int width,
        const int height,
        const float near_plane,
        const float far_plane,
        const float scale_modifier,
        const bool to_chw,
        const bool use_median_depth,
        const bool blur_periphery);
}
