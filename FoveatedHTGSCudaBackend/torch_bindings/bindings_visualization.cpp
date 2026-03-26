#include <torch/extension.h>
#include "visualization_api.h"

namespace visualization_api = htgs_foveated::visualization;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // unified rasterization api
    m.def("visualize_gaze", &visualization_api::visualize_gaze_wrapper);
    m.def("visualize_tile_boundaries", &visualization_api::visualize_tile_boundaries_wrapper);
}
