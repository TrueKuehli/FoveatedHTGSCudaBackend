#pragma once

#include "helper_math.h"
#include "utils/rasterization_utils.h"

namespace htgs_foveated::rasterization {

    void inference(
        const Buffers& buffers,
        const float3* positions,
        const float3* scales,
        const float4* rotations,
        const float* opacities,
        const float3* sh_0,
        const float3* sh_rest,
        const BackgroundModel& background_model,
        const Pose& pose,
        const Intrinsics& intrinsics,
        const Masks& masks,
        float* image,
        float* image_final,
        const int K,
        const int n_primitives,
        const int active_sh_bases,
        const int total_sh_bases,
        const float near_plane,
        const float far_plane,
        const float scale_modifier,
        const bool to_chw,
        const bool blur_periphery,
        const bool anti_aliasing);

}
