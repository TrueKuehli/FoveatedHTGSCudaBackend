#pragma once

#include <torch/extension.h>


namespace htgs_foveated::rasterization {
    std::tuple<torch::Tensor, torch::Tensor> inference_stereo_wrapper(
        const torch::Tensor& positions,
        const torch::Tensor& scales,
        const torch::Tensor& rotations,
        const torch::Tensor& opacities,
        const torch::Tensor& sh_0,
        const torch::Tensor& sh_rest,

        const torch::Tensor& M_left,
        const torch::Tensor& VPM_left,
        const torch::Tensor& VPR_inv_left,
        const torch::Tensor& cam_position_left,
        const torch::Tensor& gaze_position_left,
        const torch::Tensor& visibility_mask_left,
        const torch::Tensor& visibility_mask_area_table_left,
        const torch::Tensor& fovea_mask_area_table_left,
        const torch::Tensor& background_model_data_left,
        const int background_model_type_left,
        const int K_left,
        const int active_sh_bases_left,
        const int width_left,
        const int height_left,
        const float focal_x_left,
        const float focal_y_left,
        const float center_x_left,
        const float center_y_left,
        const float near_plane_left,
        const float far_plane_left,
        const float scale_modifier_left,

        const torch::Tensor& M_right,
        const torch::Tensor& VPM_right,
        const torch::Tensor& VPR_inv_right,
        const torch::Tensor& cam_position_right,
        const torch::Tensor& gaze_position_right,
        const torch::Tensor& visibility_mask_right,
        const torch::Tensor& visibility_mask_area_table_right,
        const torch::Tensor& fovea_mask_area_table_right,
        const torch::Tensor& background_model_data_right,
        const int background_model_type_right,
        const int K_right,
        const int active_sh_bases_right,
        const int width_right,
        const int height_right,
        const float focal_x_right,
        const float focal_y_right,
        const float center_x_right,
        const float center_y_right,
        const float near_plane_right,
        const float far_plane_right,
        const float scale_modifier_right,

        const bool to_chw,
        const bool blur_periphery,
        const bool anti_aliasing);
}
