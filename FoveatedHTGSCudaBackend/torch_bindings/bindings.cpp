#include <torch/extension.h>
#include "rasterization_api.h"

namespace rasterization_api = htgs::rasterization;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // unified rasterization api
    m.def("render", &rasterization_api::inference_wrapper);
}
