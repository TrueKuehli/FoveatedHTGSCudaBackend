#include "inference_stereo_api.h"
#include "inference_stereo.h"

#include "helper_math.h"
#include "utils/rasterization_utils.h"
#include "utils/torch_utils.h"
#include <torch/extension.h>
#include <stdexcept>
#include <functional>
#include <tuple>


std::tuple<torch::Tensor, torch::Tensor> htgs_foveated::rasterization::inference_stereo_wrapper(
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
        const bool anti_aliasing)
{
    // Pack pose and intrinsics for left and right eye
    const Pose pose_left = {
        reinterpret_cast<const float4*>(M_left.contiguous().data_ptr<float>()),
        reinterpret_cast<const float4*>(VPM_left.contiguous().data_ptr<float>()),
        reinterpret_cast<const float4*>(VPR_inv_left.contiguous().data_ptr<float>()),
        reinterpret_cast<const float3*>(cam_position_left.contiguous().data_ptr<float>()),
        reinterpret_cast<const float2*>(gaze_position_left.contiguous().data_ptr<float>())
    };
    const Pose pose_right = {
        reinterpret_cast<const float4*>(M_right.contiguous().data_ptr<float>()),
        reinterpret_cast<const float4*>(VPM_right.contiguous().data_ptr<float>()),
        reinterpret_cast<const float4*>(VPR_inv_right.contiguous().data_ptr<float>()),
        reinterpret_cast<const float3*>(cam_position_right.contiguous().data_ptr<float>()),
        reinterpret_cast<const float2*>(gaze_position_right.contiguous().data_ptr<float>())
    };
    const Intrinsics intrinsics_left = {
        width_left,
        height_left,
        focal_x_left,
        focal_y_left,
        center_x_left,
        center_y_left
    };
    const Intrinsics intrinsics_right = {
        width_right,
        height_right,
        focal_x_right,
        focal_y_right,
        center_x_right,
        center_y_right
    };
    const Masks masks_left = {
        visibility_mask_left.data_ptr<uint>(),
        visibility_mask_area_table_left.data_ptr<uint>(),
        fovea_mask_area_table_left.data_ptr<uint>()
    };
    const Masks masks_right = {
        visibility_mask_right.data_ptr<uint>(),
        visibility_mask_area_table_right.data_ptr<uint>(),
        fovea_mask_area_table_right.data_ptr<uint>()
    };

    // Assume background model type is the same for both eyes
    const BackgroundModel bg_model = {
        background_model_data_left.data_ptr<float>(),
        static_cast<BackgroundModelType>(background_model_type_left)
    };
    (void) background_model_data_right; // Unused
    (void) background_model_type_right; // Unused

    const int n_primitives = positions.size(0);
    const int total_sh_bases = sh_rest.size(1);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor image_left = to_chw ? torch::empty({3, height_left, width_left}, float_options) : torch::empty({height_left, width_left, 3}, float_options);
    torch::Tensor image_right = to_chw ? torch::empty({3, height_right, width_right}, float_options) : torch::empty({height_right, width_right, 3}, float_options);
    // When blurring the periphery, we need an extra image buffer
    torch::Tensor image_temp_left = blur_periphery ?
            (to_chw ? torch::empty({3, height_left, width_left}, float_options) : torch::empty({height_left, width_left, 3}, float_options))
            : torch::empty({0}, float_options);
    torch::Tensor image_temp_right = blur_periphery ?
            (to_chw ? torch::empty({3, height_right, width_right}, float_options) : torch::empty({height_right, width_right, 3}, float_options))
            : torch::empty({0}, float_options);

    torch::Tensor buffers[8] = {
        torch::empty({0}, byte_options), // per_primitive_buffers_left
        torch::empty({0}, byte_options), // per_tile_buffers_left
        torch::empty({0}, byte_options), // per_subtile_buffers_left
        torch::empty({0}, byte_options), // per_instance_buffers_left

        torch::empty({0}, byte_options), // per_primitive_buffers_right
        torch::empty({0}, byte_options), // per_tile_buffers_right
        torch::empty({0}, byte_options), // per_subtile_buffers_right
        torch::empty({0}, byte_options)  // per_instance_buffers_right
    };
    const Buffers buffers_left = {
        resize_function_wrapper(buffers[0]),
        resize_function_wrapper(buffers[1]),
        resize_function_wrapper(buffers[2]),
        resize_function_wrapper(buffers[3])
    };
    const Buffers buffers_right = {
        resize_function_wrapper(buffers[4]),
        resize_function_wrapper(buffers[5]),
        resize_function_wrapper(buffers[6]),
        resize_function_wrapper(buffers[7])
    };

    inference_stereo(
        buffers_left,
        buffers_right,
        reinterpret_cast<const float3*>(positions.contiguous().data_ptr<float>()),
        reinterpret_cast<const float3*>(scales.contiguous().data_ptr<float>()),
        reinterpret_cast<const float4*>(rotations.contiguous().data_ptr<float>()),
        opacities.contiguous().data_ptr<float>(),
        reinterpret_cast<const float3*>(sh_0.contiguous().data_ptr<float>()),
        reinterpret_cast<const float3*>(sh_rest.contiguous().data_ptr<float>()),
        bg_model,
        pose_left,
        pose_right,
        intrinsics_left,
        intrinsics_right,
        masks_left,
        masks_right,
        blur_periphery ? image_temp_left.data_ptr<float>() : image_left.data_ptr<float>(),
        image_left.data_ptr<float>(),
        blur_periphery ? image_temp_right.data_ptr<float>() : image_right.data_ptr<float>(),
        image_right.data_ptr<float>(),
        K_left,
        n_primitives,
        active_sh_bases_left,
        total_sh_bases,
        near_plane_left,
        far_plane_left,
        scale_modifier_left,
        to_chw,
        blur_periphery,
        anti_aliasing);

    return {image_left, image_right};
}
