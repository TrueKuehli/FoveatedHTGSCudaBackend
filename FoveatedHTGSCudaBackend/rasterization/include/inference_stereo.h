#pragma once

#include "helper_math.h"
#include "utils/rasterization_utils.h"
#include <functional>

namespace htgs_foveated::rasterization {

    void inference_stereo(
        const Buffers& buffers_left,
        const Buffers& buffers_right,
        const float3* positions,
        const float3* scales,
        const float4* rotations,
        const float* opacities,
        const float3* sh_0,
        const float3* sh_rest,
        const BackgroundModel& background_model,
        const Pose& pose_left,
        const Pose& pose_right,
        const Intrinsics& intrinsics_left,
        const Intrinsics& intrinsics_right,
        const Masks& masks_left,
        const Masks& masks_right,
        float* image_left,
        float* image_left_final,
        float* image_right,
        float* image_right_final,
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
