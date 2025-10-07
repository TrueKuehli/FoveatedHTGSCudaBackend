#include "rasterization_api.h"
#include "hybrid_blend/forward.h"
#include "hybrid_blend/backward.h"
#include "hybrid_blend/inference.h"
#include "hybrid_blend/update_max_weights.h"

#include "torch_utils.h"
#include "helper_math.h"
#include <torch/extension.h>
#include <stdexcept>
#include <functional>
#include <tuple>


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int>
htgs::rasterization::forward_wrapper(
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
    const int K,
    const int active_sh_bases,
    const int width,
    const int height,
    const float near_plane,
    const float far_plane,
    const float scale_modifier
) {
    const int n_primitives = positions.size(0);
    const int total_sh_bases = sh_rest.size(1);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor image = torch::empty({3, height, width}, float_options);
    torch::Tensor per_primitive_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_tile_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_instance_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_pixel_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> per_primitive_buffers_func = resize_function_wrapper(per_primitive_buffers);
    const std::function<char*(size_t)> per_tile_buffers_func = resize_function_wrapper(per_tile_buffers);
    const std::function<char*(size_t)> per_instance_buffers_func = resize_function_wrapper(per_instance_buffers);
    const std::function<char*(size_t)> per_pixel_buffers_func = resize_function_wrapper(per_pixel_buffers);

    std::pair<int, int> buffer_state;
    buffer_state = hybrid_blend::forward(
        per_primitive_buffers_func,
        per_tile_buffers_func,
        per_instance_buffers_func,
        per_pixel_buffers_func,
        reinterpret_cast<float3*>(positions.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(scales.contiguous().data_ptr<float>()),
        reinterpret_cast<float4*>(rotations.contiguous().data_ptr<float>()),
        opacities.contiguous().data_ptr<float>(),
        reinterpret_cast<float3*>(sh_0.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(sh_rest.contiguous().data_ptr<float>()),
        reinterpret_cast<float4*>(M.contiguous().data_ptr<float>()),
        reinterpret_cast<float4*>(VPM.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<float2*>(gaze_position.contiguous().data_ptr<float>()),
        image.data_ptr<float>(),
        K,
        n_primitives,
        active_sh_bases,
        total_sh_bases,
        width,
        height,
        near_plane,
        far_plane
    );

    return {image, per_primitive_buffers, per_tile_buffers, per_instance_buffers, per_pixel_buffers, buffer_state.first, buffer_state.second};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
htgs::rasterization::backward_wrapper(
    torch::Tensor& densification_info,
    const torch::Tensor& grad_image,
    const torch::Tensor& image,
    const torch::Tensor& positions,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& sh_rest,
    const torch::Tensor& per_primitive_buffers,
    const torch::Tensor& per_tile_buffers,
    const torch::Tensor& per_instance_buffers,
    const torch::Tensor& per_pixel_buffers,
    const torch::Tensor& M,
    const torch::Tensor& VPM,
    const torch::Tensor& cam_position,
    const torch::Tensor& gaze_position,
    const int K,
    const int active_sh_bases,
    const int width,
    const int height,
    const float near_plane,
    const float far_plane,
    const float scale_modifier,
    const int n_instances,
    const int instance_primitive_indices_selector,
    const bool use_distance_scaling
) {
    const int n_primitives = positions.size(0);
    const int total_sh_bases = sh_rest.size(1);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    torch::Tensor grad_positions = torch::zeros({n_primitives, 3}, float_options);
    torch::Tensor grad_scales = torch::zeros({n_primitives, 3}, float_options);
    torch::Tensor grad_rotations = torch::zeros({n_primitives, 4}, float_options);
    torch::Tensor grad_opacities = torch::zeros({n_primitives, 1}, float_options);
    torch::Tensor grad_sh_0 = torch::zeros({n_primitives, 1, 3}, float_options);
    torch::Tensor grad_sh_rest = torch::zeros({n_primitives, total_sh_bases, 3}, float_options);
    torch::Tensor grad_VPMT = torch::zeros({n_primitives * 9}, float_options);

    const bool update_densification_info = densification_info.size(0) > 0;
    const bool compute_abs_grad = densification_info.size(0) == 3;
    const bool requires_densification_info_helper = update_densification_info && compute_abs_grad;
    torch::Tensor densification_info_helper;
    if (requires_densification_info_helper) densification_info_helper = torch::zeros({n_primitives, 1}, float_options);

    hybrid_blend::backward(
        grad_image.contiguous().data_ptr<float>(),
        reinterpret_cast<float3*>(positions.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(scales.contiguous().data_ptr<float>()),
        reinterpret_cast<float4*>(rotations.contiguous().data_ptr<float>()),
        opacities.contiguous().data_ptr<float>(),
        reinterpret_cast<float3*>(sh_rest.contiguous().data_ptr<float>()),
        use_distance_scaling ? reinterpret_cast<float4*>(M.contiguous().data_ptr<float>()) : nullptr,
        reinterpret_cast<float4*>(VPM.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<char*>(per_primitive_buffers.data_ptr()),
        reinterpret_cast<char*>(per_tile_buffers.data_ptr()),
        reinterpret_cast<char*>(per_instance_buffers.data_ptr()),
        reinterpret_cast<char*>(per_pixel_buffers.data_ptr()),
        reinterpret_cast<float3*>(grad_positions.data_ptr<float>()),
        reinterpret_cast<float3*>(grad_scales.data_ptr<float>()),
        reinterpret_cast<float4*>(grad_rotations.data_ptr<float>()),
        grad_opacities.data_ptr<float>(),
        reinterpret_cast<float3*>(grad_sh_0.data_ptr<float>()),
        reinterpret_cast<float3*>(grad_sh_rest.data_ptr<float>()),
        grad_VPMT.data_ptr<float>(),
        update_densification_info ? densification_info.data_ptr<float>() : nullptr,
        requires_densification_info_helper ? densification_info_helper.data_ptr<float>() : nullptr,
        K,
        n_primitives,
        active_sh_bases,
        total_sh_bases,
        width,
        height,
        n_instances,
        instance_primitive_indices_selector,
        use_distance_scaling
    );

    return {grad_positions, grad_scales, grad_rotations, grad_opacities, grad_sh_0, grad_sh_rest};
}

std::tuple<torch::Tensor, torch::Tensor>
htgs::rasterization::inference_wrapper(
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
    const int K,
    const int active_sh_bases,
    const int width,
    const int height,
    const float near_plane,
    const float far_plane,
    const float scale_modifier,
    const bool to_chw,
    const bool use_median_depth
) {
    const int n_primitives = positions.size(0);
    const int total_sh_bases = sh_rest.size(1);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor image = to_chw ? torch::empty({3, height, width}, float_options) : torch::empty({height, width, 3}, float_options);
    torch::Tensor depth = torch::empty({height, width}, float_options);
    torch::Tensor per_primitive_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_tile_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_instance_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> per_primitive_buffers_func = resize_function_wrapper(per_primitive_buffers);
    const std::function<char*(size_t)> per_tile_buffers_func = resize_function_wrapper(per_tile_buffers);
    const std::function<char*(size_t)> per_instance_buffers_func = resize_function_wrapper(per_instance_buffers);

    hybrid_blend::inference(
        per_primitive_buffers_func,
        per_tile_buffers_func,
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
        depth.data_ptr<float>(),
        K,
        n_primitives,
        active_sh_bases,
        total_sh_bases,
        width,
        height,
        near_plane,
        far_plane,
        scale_modifier,
        to_chw,
        use_median_depth
    );

    return {image, depth};
}

void htgs::rasterization::update_max_weights_wrapper(
    torch::Tensor& max_weights,
    const torch::Tensor& positions,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& M,
    const torch::Tensor& VPM,
    const torch::Tensor& cam_position,
    const torch::Tensor& gaze_position,
    const int K,
    const int active_sh_bases,
    const int width,
    const int height,
    const float near_plane,
    const float far_plane,
    const float scale_modifier,
    const float weight_threshold
) {
    const int n_primitives = positions.size(0);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor per_primitive_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_tile_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_instance_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> per_primitive_buffers_func = resize_function_wrapper(per_primitive_buffers);
    const std::function<char*(size_t)> per_tile_buffers_func = resize_function_wrapper(per_tile_buffers);
    const std::function<char*(size_t)> per_instance_buffers_func = resize_function_wrapper(per_instance_buffers);

    hybrid_blend::update_max_weights(
        per_primitive_buffers_func,
        per_tile_buffers_func,
        per_instance_buffers_func,
        reinterpret_cast<const float3*>(positions.contiguous().data_ptr<float>()),
        reinterpret_cast<const float3*>(scales.contiguous().data_ptr<float>()),
        reinterpret_cast<const float4*>(rotations.contiguous().data_ptr<float>()),
        opacities.contiguous().data_ptr<float>(),
        reinterpret_cast<const float4*>(M.contiguous().data_ptr<float>()),
        reinterpret_cast<const float4*>(VPM.contiguous().data_ptr<float>()),
        max_weights.data_ptr<float>(),
        K,
        n_primitives,
        width,
        height,
        near_plane,
        far_plane,
        weight_threshold
    );
}
