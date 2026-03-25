#include <torch/extension.h>
#include "inference_api.h"

namespace rasterization_api = htgs_foveated::rasterization;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // unified rasterization api
    m.def("render", &rasterization_api::inference_wrapper);
    m.def("visualize_gaze", &rasterization_api::visualize_gaze_wrapper);
    m.def("visualize_tile_boundaries", &rasterization_api::visualize_tile_boundaries_wrapper);
}
