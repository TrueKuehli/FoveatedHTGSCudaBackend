#pragma once

// Windows needs some special princess treatment
#ifdef _WIN32
    #ifndef M_PIf
    #define M_PIf 3.14159265358979323846f
    #endif
#endif

#include "helper_math.h"
#include <cstdint>

#ifdef _WIN32
#include <climits>
#define __UINT32_MAX__ UINT32_MAX
#endif


constexpr const uint8_t MAX_CAMERAS = 2;
__device__ __constant__ float4 c_M[MAX_CAMERAS][3];
__device__ __constant__ float4 c_VPM[MAX_CAMERAS][4];
__device__ __constant__ float4 c_VPR_inv[MAX_CAMERAS][4];
__device__ __constant__ float3 c_cam_position[MAX_CAMERAS];
__device__ __constant__ float2 c_gaze_position_cuda[MAX_CAMERAS];
__device__ __constant__ float3 c_background_sh_coeff[16];

struct Mat3x3 {
    float r11, r12, r13;
    float r21, r22, r23;
    float r31, r32, r33;
};

template<typename T>
__device__ void swap(
    T& a,
    T& b)
{
    T temp = a;
    a = b;
    b = temp;
}

template <uint radius>
__forceinline__ __device__ bool is_in_fovea(
    const uint tile_idx,
    const uint grid_width,
    const float2 gaze_position_tiles)
{
    constexpr float radius_sq = static_cast<float>(radius * radius);
    const float2 tile_coords = make_float2(
        tile_idx % grid_width + 0.5f,
        tile_idx / grid_width + 0.5f
    );
    const float2 to_gaze = tile_coords - gaze_position_tiles;
    const float squared_distance_to_gaze = dot(to_gaze, to_gaze);

    return squared_distance_to_gaze < radius_sq;
}


template <uint radius>
__forceinline__ __device__ bool is_in_fovea(
    const int2 tile_coords,
    const uint grid_width,
    const float2 gaze_position_tiles)
{
    constexpr float radius_sq = static_cast<float>(radius * radius);
    const float2 to_gaze = make_float2(tile_coords.x + 0.5f, tile_coords.y + 0.5f) - gaze_position_tiles;
    const float squared_distance_to_gaze = dot(to_gaze, to_gaze);

    return squared_distance_to_gaze < radius_sq;
}


__forceinline__ __device__ Mat3x3 convert_quaternion_to_rotation_matrix(
    const float4& quaternion)
{
    auto [r, x, y, z] = quaternion;
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float rx = r * x, ry = r * y, rz = r * z;
    return {
        1.0f - 2.0f * (yy + zz), 2.0f * (xy - rz), 2.0f * (xz + ry),
        2.0f * (xy + rz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz - rx),
        2.0f * (xz - ry), 2.0f * (yz + rx), 1.0f - 2.0f * (xx + yy)
    };
}


template<uint8_t cam_idx>
__forceinline__ __device__ bool transform_and_cull(
    const float3* scales,
    const float4* rotations,
    const float3& position_world,
    const float4& M3,
    uint& n_touched_tiles,
    uint4& screen_bounds,
    float3& u,
    float3& v,
    float3& w,
    float4& VPMT1,
    float4& VPMT2,
    float4& VPMT4,
    float& z,
    float& opacity,
    const uint* render_mask_area_table,
    const uint* fovea_mask_area_table,
    const uint primitive_idx,
    const uint grid_width,
    const uint grid_height,
    const uint tile_width,
    const uint tile_height,
    const uint foveation_radius_tiles,
    const int2 gaze_position,
    const float near_plane,
    const float far_plane,
    const float min_alpha_threshold_rcp,
    const float scale_modifier)
{
    // early near_plane/far_plane plane culling
    z = dot(make_float3(M3), position_world) + M3.w;
    if (z < near_plane || z > far_plane) return true;

    // load scale, rotation, and opacity
    const float3 scale = scales[primitive_idx];
    const float4 quaternion = rotations[primitive_idx];
    const Mat3x3 R = convert_quaternion_to_rotation_matrix(quaternion);

    // compute screen-space bounding box
    u = make_float3(R.r11 * scale.x, R.r21 * scale.x, R.r31 * scale.x) * scale_modifier;
    v = make_float3(R.r12 * scale.y, R.r22 * scale.y, R.r32 * scale.y) * scale_modifier;
    w = make_float3(R.r13 * scale.z, R.r23 * scale.z, R.r33 * scale.z) * scale_modifier;
    const float4 VPM4 = c_VPM[cam_idx][3];
    VPMT4 = make_float4(dot(make_float3(VPM4), u), dot(make_float3(VPM4), v), dot(make_float3(VPM4), w), dot(make_float3(VPM4), position_world) + VPM4.w);
    // tight cutoff for the used opacity threshold
    const float rho_cutoff = 2.0f * logf(opacity * min_alpha_threshold_rcp);
    const float4 d = make_float4(rho_cutoff, rho_cutoff, rho_cutoff, -1.0f);
    const float s = dot(d, VPMT4 * VPMT4);
    if (s == 0.0f) return true;
    const float4 f = (1.0f / s) * d;
    // start with z-extent in screen-space for exact near_plane/far_plane plane culling
    const float4 VPM3 = c_VPM[cam_idx][2];
    const float4 VPMT3 = make_float4(dot(make_float3(VPM3), u), dot(make_float3(VPM3), v), dot(make_float3(VPM3), w), dot(make_float3(VPM3), position_world) + VPM3.w);
    const float center_z = dot(f, VPMT3 * VPMT4);
    const float extent_z = sqrtf(fmaxf(center_z * center_z - dot(f, VPMT3 * VPMT3), 0.0f));
    const float z_min = center_z - extent_z;
    const float z_max = center_z + extent_z;
    if (z_min < -1.0f || z_max > 1.0f) return true;
    // now x/y-extent of the screen-space bounding box
    const float4 VPM1 = c_VPM[cam_idx][0];
    VPMT1 = make_float4(dot(make_float3(VPM1), u), dot(make_float3(VPM1), v), dot(make_float3(VPM1), w), dot(make_float3(VPM1), position_world) + VPM1.w);
    const float center_x = dot(f, VPMT1 * VPMT4);
    const float extent_x = sqrtf(fmaxf(center_x * center_x - dot(f, VPMT1 * VPMT1), 0.0f));
    const float4 VPM2 = c_VPM[cam_idx][1];
    VPMT2 = make_float4(dot(make_float3(VPM2), u), dot(make_float3(VPM2), v), dot(make_float3(VPM2), w), dot(make_float3(VPM2), position_world) + VPM2.w);
    const float center_y = dot(f, VPMT2 * VPMT4);
    const float extent_y = sqrtf(fmaxf(center_y * center_y - dot(f, VPMT2 * VPMT2), 0.0f));

    // compute screen-space bounding box in tile coordinates (+0.5 to account for half-pixel shift in V)
    screen_bounds = make_uint4(
        min(grid_width, static_cast<uint>(max(0, __float2int_rd((center_x - extent_x + 0.5f) / tile_width)))), // x_min
        min(grid_width, static_cast<uint>(max(0, __float2int_ru((center_x + extent_x + 0.5f) / tile_width)))), // x_max
        min(grid_height, static_cast<uint>(max(0, __float2int_rd((center_y - extent_y + 0.5f) / tile_height)))), // y_min
        min(grid_height, static_cast<uint>(max(0, __float2int_ru((center_y + extent_y + 0.5f) / tile_height)))) // y_max
    );

    const int foveation_diameter_tiles = 2 * foveation_radius_tiles;
    const int2 mask_top_left = make_int2(
        gaze_position.x - foveation_radius_tiles,
        gaze_position.y - foveation_radius_tiles
    );
    const uint4 foveation_table_bounds = make_uint4(
        min(foveation_diameter_tiles, max(0, static_cast<int>(screen_bounds.x) - mask_top_left.x)), // x_min
        min(foveation_diameter_tiles, max(0, static_cast<int>(screen_bounds.y) - mask_top_left.x)), // x_max
        min(foveation_diameter_tiles, max(0, static_cast<int>(screen_bounds.z) - mask_top_left.y)), // y_min
        min(foveation_diameter_tiles, max(0, static_cast<int>(screen_bounds.w) - mask_top_left.y))  // y_max
    );

    // get number of potentially influenced tiles via summed area table
    // assume bounding box A - B
    //                     |   |
    //                     C - D
    const uint4 area_table_indices = make_uint4(
        screen_bounds.x + (grid_width + 1) * screen_bounds.z,  // A
        screen_bounds.y + (grid_width + 1) * screen_bounds.z,  // B
        screen_bounds.x + (grid_width + 1) * screen_bounds.w,  // C
        screen_bounds.y + (grid_width + 1) * screen_bounds.w   // D
    );

    // compute number of potentially influenced tiles
    // n_tiles = area(D) + area(A) - area(B) - area(C)
    // This may overestimate the actual amount (as tiles masked by the render mask are not excluded from the fovea mask)
    //   but this is acceptable as it only leads to some redundant work
    n_touched_tiles = render_mask_area_table[area_table_indices.w]
                    + render_mask_area_table[area_table_indices.x]
                    - render_mask_area_table[area_table_indices.y]
                    - render_mask_area_table[area_table_indices.z];
    if (foveation_table_bounds.y != foveation_table_bounds.x && foveation_table_bounds.w != foveation_table_bounds.z) {
        const uint4 fovea_table_indices = make_uint4(
            foveation_table_bounds.x + (foveation_diameter_tiles + 1) * foveation_table_bounds.z, // A
            foveation_table_bounds.y + (foveation_diameter_tiles + 1) * foveation_table_bounds.z, // B
            foveation_table_bounds.x + (foveation_diameter_tiles + 1) * foveation_table_bounds.w, // C
            foveation_table_bounds.y + (foveation_diameter_tiles + 1) * foveation_table_bounds.w  // D
        );
        n_touched_tiles += fovea_mask_area_table[fovea_table_indices.w]
                         + fovea_mask_area_table[fovea_table_indices.x]
                         - fovea_mask_area_table[fovea_table_indices.y]
                         - fovea_mask_area_table[fovea_table_indices.z];
    }

    return n_touched_tiles == 0;
}
