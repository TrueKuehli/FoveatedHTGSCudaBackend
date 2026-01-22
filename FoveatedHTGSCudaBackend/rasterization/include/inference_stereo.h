#pragma once

#include "helper_math.h"
#include "utils/rasterization_utils.h"
#include <functional>

namespace htgs_foveated::rasterization {

    struct Buffers {
        std::function<char* (size_t)> per_primitive_buffers_func;
        std::function<char* (size_t)> per_tile_buffers_func;
        std::function<char* (size_t)> per_subtile_buffers_func;
        std::function<char* (size_t)> per_instance_buffers_func;
    };

    struct Pose {
        const float4* M;
        const float4* VPM;
        const float4* VPR_inv;
        const float3* cam_position;
        const float2* gaze_position;
    };

    struct Intrinsics {
        const int width;
        const int height;
        const float focal_x;
        const float focal_y;
        const float center_x;
        const float center_y;
    };
    
    struct Masks {
        const uint* render_mask;
        const uint* render_mask_area_table;
        const uint* fovea_mask_area_table;
    };

    struct BackgroundModel {
        const float* data;
        const BackgroundModelType type;
    };


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
