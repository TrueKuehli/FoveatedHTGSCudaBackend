#pragma once

#include "helper_math.h"
#include "rasterization_utils.h"
#include <functional>

namespace htgs_foveated::rasterization::hybrid_blend {

    void fast_inference(
        std::function<char* (size_t)> per_primitive_buffers_func,
        std::function<char* (size_t)> per_tile_buffers_func,
        std::function<char* (size_t)> per_subtile_buffers_func,
        std::function<char* (size_t)> per_instance_buffers_func,
        const float3* positions,
        const float3* scales,
        const float4* rotations,
        const float* opacities,
        const float3* sh_0,
        const float3* sh_rest,
        const float4* M,
        const float4* VPM,
        const float4* VPR_inv,
        const float3* cam_position,
        const float2* gaze_position,
        float* image,
        float* image_final,
        const uint* render_mask,
        const uint* render_mask_area_table,
        const uint* fovea_mask_area_table,
        const float* background_model_data,
        const BackgroundModelType background_model_type,
        const PeripheryInterpolationMode periphery_mode,
        const int K,
        const int n_primitives,
        const int active_sh_bases,
        const int total_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float near_plane,
        const float far_plane,
        const float scale_modifier,
        const bool to_chw,
        const bool blur_periphery,
        const bool anti_aliasing);

}
