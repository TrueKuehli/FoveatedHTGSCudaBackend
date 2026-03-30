#pragma once

#include <functional>


enum class BackgroundModelType {
    NONE = 0,
    SH = 1,
    TEXTURE = 2,
};

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


inline __host__ int extract_end_bit(
    uint n)
{
    int leading_zeros = 0;
    if ((n & 0xffff0000) == 0) { leading_zeros += 16; n <<= 16; }
    if ((n & 0xff000000) == 0) { leading_zeros += 8; n <<= 8; }
    if ((n & 0xf0000000) == 0) { leading_zeros += 4; n <<= 4; }
    if ((n & 0xc0000000) == 0) { leading_zeros += 2; n <<= 2; }
    if ((n & 0x80000000) == 0) { leading_zeros += 1; }
    return 32 - leading_zeros;
}
