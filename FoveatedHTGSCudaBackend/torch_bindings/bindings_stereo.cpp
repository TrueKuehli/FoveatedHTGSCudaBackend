#include <torch/extension.h>
#include "inference_stereo_api.h"

namespace rasterization_api = htgs_foveated::rasterization;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // unified rasterization api
    m.def("render", &rasterization_api::inference_stereo_wrapper);
}
