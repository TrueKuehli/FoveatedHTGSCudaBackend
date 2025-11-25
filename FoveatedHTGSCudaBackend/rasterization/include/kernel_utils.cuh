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

__device__ __constant__ float4 c_M3;
__device__ __constant__ float4 c_VPM[4];
__device__ __constant__ float4 c_VPR_inv[4];
__device__ __constant__ float3 c_cam_position;
__device__ __constant__ float2 c_gaze_position_cuda;
__device__ __constant__ float3 c_background_sh_coeff[16];
__device__ __constant__ uint32_t c_render_mask[4096];  // Sufficient for ~362x362 tiles (total 131,072)

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
    const uint2 gaze_position_tiles)
{
    constexpr uint radius_sq = radius * radius;
    const int2 tile_coords = make_int2(
        tile_idx % grid_width,
        tile_idx / grid_width
    );
    const int2 to_gaze = tile_coords - make_int2(gaze_position_tiles.x, gaze_position_tiles.y);
    const int squared_distance_to_gaze = dot(to_gaze, to_gaze);

    return squared_distance_to_gaze < radius_sq;
}


template <uint radius>
__forceinline__ __device__ bool is_in_fovea(
    const int2 tile_coords,
    const uint grid_width,
    const int2 gaze_position_tiles)
{
    constexpr uint radius_sq = radius * radius;
    const int2 to_gaze = tile_coords - gaze_position_tiles;
    const int squared_distance_to_gaze = dot(to_gaze, to_gaze);

    return squared_distance_to_gaze < radius_sq;
}


__forceinline__ __device__ Mat3x3 convert_quaterion_to_rotation_matrix(
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


template<bool anti_aliasing>
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
    const uint2 gaze_position,
    const float focal_x,
    const float focal_y,
    const float near_plane,
    const float far_plane,
    const float min_alpha_threshold_rcp,
    const float scale_modifier,
    const float aa_kernel_size)
{
    // early near_plane/far_plane plane culling
    z = dot(make_float3(M3), position_world) + M3.w;
    if (z < near_plane || z > far_plane) return true;

    // load scale, rotation, and opacity
    const float3 scale = scales[primitive_idx];
    const float4 quaternion = rotations[primitive_idx];
    const Mat3x3 R = convert_quaterion_to_rotation_matrix(quaternion);

    // Calculate dilated scale for anti-aliasing (if enabled)
    float3 scale_dilated;
    if constexpr (anti_aliasing) {
        const float focal = 0.5f * (focal_x + focal_y);
        const float mip_filter_scale = fmaxf(
            (z / focal) * (z / focal) * aa_kernel_size,
            0  // filter_3d * filter_3d  // TODO: ???
        );
        scale_dilated = make_float3(
            sqrtf(scale.x * scale.x + mip_filter_scale),
            sqrtf(scale.y * scale.y + mip_filter_scale),
            sqrtf(scale.z * scale.z + mip_filter_scale)
        );

        const float3 view_dir_world = normalize(position_world - c_cam_position);
        const float3 view_dir = make_float3(
            R.r11 * view_dir_world.x + R.r21 * view_dir_world.y + R.r31 * view_dir_world.z,
            R.r12 * view_dir_world.x + R.r22 * view_dir_world.y + R.r32 * view_dir_world.z,
            R.r13 * view_dir_world.x + R.r23 * view_dir_world.y + R.r33 * view_dir_world.z
        );
        const float3 r = view_dir * view_dir;

        const float3 s_2 = scale * scale;
        const float3 s_dil_2 = scale_dilated * scale_dilated;
        const float det_mul_ray_var = dot(r, make_float3(s_2.y * s_2.z, s_2.z * s_2.x, s_2.x * s_2.y));
        const float det_mul_ray_var_dil = dot(r, make_float3(s_dil_2.y * s_dil_2.z, s_dil_2.z * s_dil_2.x, s_dil_2.x * s_dil_2.y));
        const float dilation_factor = sqrtf(det_mul_ray_var / det_mul_ray_var_dil);
        opacity *= dilation_factor;
    } else {
        scale_dilated = scale;
    }

    // compute screen-space bounding box
    u = make_float3(R.r11 * scale_dilated.x, R.r21 * scale_dilated.x, R.r31 * scale_dilated.x) * scale_modifier;
    v = make_float3(R.r12 * scale_dilated.y, R.r22 * scale_dilated.y, R.r32 * scale_dilated.y) * scale_modifier;
    w = make_float3(R.r13 * scale_dilated.z, R.r23 * scale_dilated.z, R.r33 * scale_dilated.z) * scale_modifier;
    const float4 VPM4 = c_VPM[3];
    VPMT4 = make_float4(dot(make_float3(VPM4), u), dot(make_float3(VPM4), v), dot(make_float3(VPM4), w), dot(make_float3(VPM4), position_world) + VPM4.w);
    // tight cutoff for the used opacity threshold
    const float rho_cutoff = 2.0f * logf(opacity * min_alpha_threshold_rcp);
    const float4 d = make_float4(rho_cutoff, rho_cutoff, rho_cutoff, -1.0f);
    const float s = dot(d, VPMT4 * VPMT4);
    if (s == 0.0f) return true;
    const float4 f = (1.0f / s) * d;
    // start with z-extent in screen-space for exact near_plane/far_plane plane culling
    const float4 VPM3 = c_VPM[2];
    const float4 VPMT3 = make_float4(dot(make_float3(VPM3), u), dot(make_float3(VPM3), v), dot(make_float3(VPM3), w), dot(make_float3(VPM3), position_world) + VPM3.w);
    const float center_z = dot(f, VPMT3 * VPMT4);
    const float extent_z = sqrtf(fmaxf(center_z * center_z - dot(f, VPMT3 * VPMT3), 0.0f));
    const float z_min = center_z - extent_z;
    const float z_max = center_z + extent_z;
    if (z_min < -1.0f || z_max > 1.0f) return true;
    // now x/y-extent of the screen-space bounding box
    const float4 VPM1 = c_VPM[0];
    VPMT1 = make_float4(dot(make_float3(VPM1), u), dot(make_float3(VPM1), v), dot(make_float3(VPM1), w), dot(make_float3(VPM1), position_world) + VPM1.w);
    const float center_x = dot(f, VPMT1 * VPMT4);
    const float extent_x = sqrtf(fmaxf(center_x * center_x - dot(f, VPMT1 * VPMT1), 0.0f));
    const float4 VPM2 = c_VPM[1];
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
        static_cast<int>(gaze_position.x) - foveation_radius_tiles,
        static_cast<int>(gaze_position.y) - foveation_radius_tiles
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
    // TODO: We could compare performance with re-calculating the summed area tables each frame
    // TODO: On modern GPUs, atomic adds are apparently very performant, so we could try instance creation that way, and compare performance
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

template <bool train_mode>
__forceinline__ __device__ float3 convert_sh_to_rgb(
    const float3* sh_0,
    const float3* sh_rest,
    [[maybe_unused]] bool* rgb_clamp_info,
    const float3& position_world,
    const uint n_primitives,
    const uint primitive_idx,
    const uint active_sh_bases,
    const uint total_sh_bases)
{
    // computation adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L340
    float3 result = 0.5f + 0.28209479177387814f * sh_0[primitive_idx];
    if (active_sh_bases > 1) {
        const float3* coefficients_ptr = sh_rest + primitive_idx * total_sh_bases;
        auto [x, y, z] = normalize(position_world - c_cam_position);
        result = result + (-0.48860251190291987f * y) * coefficients_ptr[0]
                        + (0.48860251190291987f * z) * coefficients_ptr[1]
                        + (-0.48860251190291987f * x) * coefficients_ptr[2];
        if (active_sh_bases > 4) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, xz = x * z, yz = y * z;
            result = result + (1.0925484305920792f * xy) * coefficients_ptr[3]
                            + (-1.0925484305920792f * yz) * coefficients_ptr[4]
                            + (0.94617469575755997f * zz - 0.31539156525251999f) * coefficients_ptr[5]
                            + (-1.0925484305920792f * xz) * coefficients_ptr[6]
                            + (0.54627421529603959f * xx - 0.54627421529603959f * yy) * coefficients_ptr[7];
            if (active_sh_bases > 9) {
                result = result + (0.59004358992664352f * y * (-3.0f * xx + yy)) * coefficients_ptr[8]
                                + (2.8906114426405538f * xy * z) * coefficients_ptr[9]
                                + (0.45704579946446572f * y * (1.0f - 5.0f * zz)) * coefficients_ptr[10]
                                + (0.3731763325901154f * z * (5.0f * zz - 3.0f)) * coefficients_ptr[11]
                                + (0.45704579946446572f * x * (1.0f - 5.0f * zz)) * coefficients_ptr[12]
                                + (1.4453057213202769f * z * (xx - yy)) * coefficients_ptr[13]
                                + (0.59004358992664352f * x * (-xx + 3.0f * yy)) * coefficients_ptr[14];
            }
        }
    }
    if constexpr (train_mode) {
        rgb_clamp_info[primitive_idx] = result.x < 0;
        rgb_clamp_info[n_primitives + primitive_idx] = result.y < 0;
        rgb_clamp_info[2 * n_primitives + primitive_idx] = result.z < 0;
    }
    return {
        fmaxf(0.0f, result.x),
        fmaxf(0.0f, result.y),
        fmaxf(0.0f, result.z)
    };
}


__forceinline__ __device__ float3 eval_sh_background_model(const float pixel_x, const float pixel_y) {
    // computation adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L340
    const float4 pixel_coords = make_float4(pixel_x, pixel_y, 1.0, 1.0);
    const float3 pixel_coords_transformed = normalize(make_float3(
        dot(c_VPR_inv[0], pixel_coords),
        dot(c_VPR_inv[1], pixel_coords),
        dot(c_VPR_inv[2], pixel_coords)
    ));

    // TODO: Test if branching is faster (since that could save some computations for any grid cells that are fully below the horizon)
    auto [x, y, z] = normalize(pixel_coords_transformed);
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    // TODO: Use FastGS SH eval code
    const float3 sh_eval = 0.28209479177387814f * c_background_sh_coeff[0]
                    + (-0.48860251190291987f * y) * c_background_sh_coeff[1]
                    + (0.48860251190291987f * z) * c_background_sh_coeff[2]
                    + (-0.48860251190291987f * x) * c_background_sh_coeff[3]
                    + (1.0925484305920792f * xy) * c_background_sh_coeff[4]
                    + (-1.0925484305920792f * yz) * c_background_sh_coeff[5]
                    + (0.94617469575755997f * zz - 0.31539156525251999f) * c_background_sh_coeff[6]
                    + (-1.0925484305920792f * xz) * c_background_sh_coeff[7]
                    + (0.54627421529603959f * xx - 0.54627421529603959f * yy) * c_background_sh_coeff[8]
                    + (0.59004358992664352f * y * (-3.0f * xx + yy)) * c_background_sh_coeff[9]
                    + (2.8906114426405538f * xy * z) * c_background_sh_coeff[10]
                    + (0.45704579946446572f * y * (1.0f - 5.0f * zz)) * c_background_sh_coeff[11]
                    + (0.3731763325901154f * z * (5.0f * zz - 3.0f)) * c_background_sh_coeff[12]
                    + (0.45704579946446572f * x * (1.0f - 5.0f * zz)) * c_background_sh_coeff[13]
                    + (1.4453057213202769f * z * (xx - yy)) * c_background_sh_coeff[14]
                    + (0.59004358992664352f * x * (-xx + 3.0f * yy)) * c_background_sh_coeff[15];
    const float3 result = make_float3(
        tanhf(sh_eval.x) * 0.5f + 0.5f,
        tanhf(sh_eval.y) * 0.5f + 0.5f,
        tanhf(sh_eval.z) * 0.5f + 0.5f
    );

    // Return black below the horizon
    return pixel_coords_transformed.y > 0 ? make_float3(0.0f, 0.0f, 0.0f) : make_float3(
        __saturatef(result.x),
        __saturatef(result.y),
        __saturatef(result.z)
    );
}


template<int width, int height>
__forceinline__ __device__ float3 eval_tex_background_model(const float pixel_x, const float pixel_y, const float* texture_data) {
    // computation adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L340
    const float4 pixel_coords = make_float4(pixel_x, pixel_y, 1.0, 1.0);
    const float3 pixel_coords_transformed = normalize(make_float3(
        dot(c_VPR_inv[0], pixel_coords),
        dot(c_VPR_inv[1], pixel_coords),
        dot(c_VPR_inv[2], pixel_coords)
    ));

    // TODO: Test if branching is faster (since that could save some computations / memory accesses for any grid cells that are fully below the horizon)
    const float2 equirectangular_coords = make_float2(
        atan2f(pixel_coords_transformed.x, pixel_coords_transformed.z) / M_PIf * 0.5f + 0.5f,
        asinf(-pixel_coords_transformed.y) / (0.5f * M_PIf) * 0.5f + 0.5f
    );

    // Bilinear interpolation
    const float tex_x = equirectangular_coords.x * width - 0.5f;
    const float tex_y = equirectangular_coords.y * height - 0.5f;

    const int x0 = __float2int_rd(tex_x);
    const int y0 = __float2int_rd(tex_y);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const float fx = tex_x - x0;
    const float fy = tex_y - y0;

    // Wrap coordinates for seamless horizontal tiling
    const int x0_wrapped = (x0 % width + width) % width;
    const int x1_wrapped = (x1 % width + width) % width;
    const int y0_clamped = max(0, min(height - 1, y0));
    const int y1_clamped = max(0, min(height - 1, y1));

    // Get texture indices for the four corners
    const int idx00 = x0_wrapped + y0_clamped * width;
    const int idx10 = x1_wrapped + y0_clamped * width;
    const int idx01 = x0_wrapped + y1_clamped * width;
    const int idx11 = x1_wrapped + y1_clamped * width;

    // Bilinear interpolation for each color channel
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    for (int channel = 0; channel < 3; ++channel) {
        const int offset = channel * width * height;
        const float c00 = texture_data[idx00 + offset];
        const float c10 = texture_data[idx10 + offset];
        const float c01 = texture_data[idx01 + offset];
        const float c11 = texture_data[idx11 + offset];

        const float c0 = c00 * (1.0f - fx) + c10 * fx;
        const float c1 = c01 * (1.0f - fx) + c11 * fx;
        const float c = c0 * (1.0f - fy) + c1 * fy;

        if (channel == 0) color.x = c;
        else if (channel == 1) color.y = c;
        else color.z = c;
    }

    // Return black below the horizon
    return pixel_coords_transformed.y > 0 ? make_float3(0.0f, 0.0f, 0.0f) : color;
}
