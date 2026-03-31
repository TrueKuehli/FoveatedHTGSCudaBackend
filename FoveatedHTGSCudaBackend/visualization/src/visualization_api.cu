#include "visualization_api.h"
#include "visualization.h"

#include "enums.h"
#include "helper_math.h"
#include "utils/torch_utils.h"
#include <functional>
#include <stdexcept>
#include <tuple>
#include <torch/extension.h>


torch::Tensor htgs_foveated::visualization::visualize_gaze_wrapper(
    const torch::Tensor& image,
    const torch::Tensor& gaze_position,
    const torch::Tensor& gaze_color,
    const uint8_t visualization_type,
    const int visualization_size,
    const bool to_chw
) {
    const int width = image.size(to_chw ? 2 : 1);
    const int height = image.size(to_chw ? 1 : 0);
    visualize_gaze_position(
        image.data_ptr<float>(),
        width,
        height,
        visualization_size,
        reinterpret_cast<const float2*>(gaze_position.contiguous().data_ptr<float>()),
        *reinterpret_cast<const float3*>(gaze_color.contiguous().data_ptr<float>()),
        static_cast<GazeVisualizationType>(visualization_type),
        to_chw
    );

    return image;
}

torch::Tensor htgs_foveated::visualization::visualize_tile_boundaries_wrapper(
    const torch::Tensor& image,
    const torch::Tensor& visibility_mask,
    const torch::Tensor& gaze_position,
    const bool to_chw
) {
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor per_tile_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_subtile_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> per_tile_buffers_func = resize_function_wrapper(per_tile_buffers);
    const std::function<char*(size_t)> per_subtile_buffers_func = resize_function_wrapper(per_subtile_buffers);

    const int width = image.size(to_chw ? 2 : 1);
    const int height = image.size(to_chw ? 1 : 0);

    visualize_tile_boundaries(
        per_tile_buffers_func,
        per_subtile_buffers_func,
        image.data_ptr<float>(),
        reinterpret_cast<const float2*>(gaze_position.contiguous().data_ptr<float>()),
        visibility_mask.data_ptr<uint>(),
        width,
        height,
        to_chw
    );

    return image;
}
