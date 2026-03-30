#include "inference_api.h"
#include "inference.h"

#include "helper_math.h"
#include "utils/rasterization_utils.h"
#include "utils/torch_utils.h"
#include <torch/extension.h>
#include <stdexcept>
#include <functional>
#include <tuple>


torch::Tensor htgs_foveated::rasterization::inference_wrapper(
    const torch::Tensor& positions,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& sh_0,
    const torch::Tensor& sh_rest,
    const torch::Tensor& M,
    const torch::Tensor& VPM,
    const torch::Tensor& VPR_inv,
    const torch::Tensor& cam_position,
    const torch::Tensor& gaze_position,
    const torch::Tensor& render_mask,
    const torch::Tensor& render_mask_area_table,
    const torch::Tensor& fovea_mask_area_table,
    const torch::Tensor& background_model_data,
    const int background_model_type,
    const int K,
    const int active_sh_bases,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const float scale_modifier,
    const bool to_chw,
    const bool blur_periphery,
    const bool anti_aliasing)
{
    // Pack structs
    const Pose pose = {
        reinterpret_cast<const float4*>(M.contiguous().data_ptr<float>()),
        reinterpret_cast<const float4*>(VPM.contiguous().data_ptr<float>()),
        reinterpret_cast<const float4*>(VPR_inv.contiguous().data_ptr<float>()),
        reinterpret_cast<const float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<const float2*>(gaze_position.contiguous().data_ptr<float>())
    };
    const Intrinsics intrinsics = {
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y
    };
    const Masks masks = {
        render_mask.data_ptr<uint>(),
        render_mask_area_table.data_ptr<uint>(),
        fovea_mask_area_table.data_ptr<uint>()
    };
    const BackgroundModel bg_model = {
        background_model_data.data_ptr<float>(),
        static_cast<BackgroundModelType>(background_model_type)
    };


    const int n_primitives = positions.size(0);
    const int total_sh_bases = sh_rest.size(1);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor image = to_chw ? torch::zeros({3, height, width}, float_options) : torch::zeros({height, width, 3}, float_options);
    // When blurring the periphery, we need an extra image buffer
    torch::Tensor image_temp = blur_periphery ?
            (to_chw ? torch::zeros({3, height, width}, float_options) : torch::zeros({height, width, 3}, float_options))
            : torch::empty({0}, float_options);

    torch::Tensor per_primitive_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_tile_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_subtile_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_instance_buffers = torch::empty({0}, byte_options);
    const Buffers buffers = {
        resize_function_wrapper(per_primitive_buffers),
        resize_function_wrapper(per_tile_buffers),
        resize_function_wrapper(per_subtile_buffers),
        resize_function_wrapper(per_instance_buffers)
    };

    inference(
        buffers,
        reinterpret_cast<const float3*>(positions.contiguous().data_ptr<float>()),
        reinterpret_cast<const float3*>(scales.contiguous().data_ptr<float>()),
        reinterpret_cast<const float4*>(rotations.contiguous().data_ptr<float>()),
        opacities.contiguous().data_ptr<float>(),
        reinterpret_cast<const float3*>(sh_0.contiguous().data_ptr<float>()),
        reinterpret_cast<const float3*>(sh_rest.contiguous().data_ptr<float>()),
        bg_model,
        pose,
        intrinsics,
        masks,
        blur_periphery ? image_temp.data_ptr<float>() : image.data_ptr<float>(),
        image.data_ptr<float>(),
        K,
        n_primitives,
        active_sh_bases,
        total_sh_bases,
        near_plane,
        far_plane,
        scale_modifier,
        to_chw,
        blur_periphery,
        anti_aliasing);

    return image;
}
