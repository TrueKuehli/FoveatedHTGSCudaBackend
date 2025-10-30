#include "fast_inference_api.h"
#include "hybrid_blend/fast_inference.h"

#include "torch_utils.h"
#include "helper_math.h"
#include "rasterization_utils.h"
#include <torch/extension.h>
#include <stdexcept>
#include <functional>


torch::Tensor htgs::rasterization::fast_inference_wrapper(
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
    const float focal_x,
    const float focal_y,
    const float near_plane,
    const float far_plane,
    const float scale_modifier,
    const bool to_chw,
    const bool blur_periphery,
    const bool anti_aliasing)
{
    const PeripheryInterpolationMode periphery_mode = static_cast<PeripheryInterpolationMode>(periphery_interpolation_mode);
    const int n_primitives = positions.size(0);
    const int total_sh_bases = sh_rest.size(1);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor image = to_chw ? torch::empty({3, height, width}, float_options) : torch::empty({height, width, 3}, float_options);
    torch::Tensor per_primitive_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_tile_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_subtile_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_instance_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> per_primitive_buffers_func = resize_function_wrapper(per_primitive_buffers);
    const std::function<char*(size_t)> per_tile_buffers_func = resize_function_wrapper(per_tile_buffers);
    const std::function<char*(size_t)> per_subtile_buffers_func = resize_function_wrapper(per_subtile_buffers);
    const std::function<char*(size_t)> per_instance_buffers_func = resize_function_wrapper(per_instance_buffers);

    // When blurring the periphery, we need an extra image buffer
    torch::Tensor image_final = blur_periphery ?
            (to_chw ? torch::empty({3, height, width}, float_options) : torch::empty({height, width, 3}, float_options))
            : torch::empty({0}, float_options);

    hybrid_blend::fast_inference(
        per_primitive_buffers_func,
        per_tile_buffers_func,
        per_subtile_buffers_func,
        per_instance_buffers_func,
        reinterpret_cast<const float3*>(positions.contiguous().data_ptr<float>()),
        reinterpret_cast<const float3*>(scales.contiguous().data_ptr<float>()),
        reinterpret_cast<const float4*>(rotations.contiguous().data_ptr<float>()),
        opacities.contiguous().data_ptr<float>(),
        reinterpret_cast<const float3*>(sh_0.contiguous().data_ptr<float>()),
        reinterpret_cast<const float3*>(sh_rest.contiguous().data_ptr<float>()),
        reinterpret_cast<const float4*>(M.contiguous().data_ptr<float>()),
        reinterpret_cast<const float4*>(VPM.contiguous().data_ptr<float>()),
        reinterpret_cast<const float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<const float2*>(gaze_position.contiguous().data_ptr<float>()),
        image.data_ptr<float>(),
        blur_periphery ? image_final.data_ptr<float>() : image.data_ptr<float>(),
        render_mask.data_ptr<uint>(),
        render_mask_area_table.data_ptr<uint>(),
        fovea_mask_area_table.data_ptr<uint>(),
        periphery_mode,
        K,
        n_primitives,
        active_sh_bases,
        total_sh_bases,
        width,
        height,
        focal_x,
        focal_y,
        near_plane,
        far_plane,
        scale_modifier,
        to_chw,
        blur_periphery,
        anti_aliasing);

    return image;
}
