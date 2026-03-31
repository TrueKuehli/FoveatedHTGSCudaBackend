#pragma once

#include "utils/kernel_utils.cuh"


namespace htgs_foveated::rasterization::kernels {
    #define DEF inline constexpr float
    // degree 0
    DEF C0 = 0.28209479177387814f;
    // degree 1
    DEF C1 = 0.48860251190291987f;
    // degree 2
    DEF C2a = 1.0925484305920792f;
    DEF C2b = 0.94617469575755997f;
    DEF C2c = 0.31539156525251999f;
    DEF C2d = 0.54627421529603959f;
    DEF C2e = 1.8923493915151202f;
    // degree 3
    DEF C3a = 0.59004358992664352f;
    DEF C3b = 1.7701307697799304f;
    DEF C3c = 2.8906114426405538f;
    DEF C3d = 0.45704579946446572f;
    DEF C3e = 2.2852289973223288f;
    DEF C3f = 1.865881662950577f;
    DEF C3g = 1.1195289977703462f;
    DEF C3h = 1.4453057213202769f;
    DEF C3i = 3.5402615395598609f;
    DEF C3j = 4.5704579946446566f;
    DEF C3k = 5.597644988851731f;
    #undef DEF

    template<uint8_t cam_idx>
    __device__ inline float3 convert_sh_to_color(
        const float3* __restrict__ sh_coefficients_0,
        const float3* __restrict__ sh_coefficients_rest,
        const float3& position,
        const uint primitive_idx,
        const uint active_sh_bases,
        const uint total_sh_bases_rest)
    {
        // computation adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L340
        float3 result = 0.5f + C0 * sh_coefficients_0[primitive_idx];
        if (active_sh_bases > 1) {
            const float3* coefficients_ptr = sh_coefficients_rest + primitive_idx * total_sh_bases_rest;
            auto [x, y, z] = normalize(position - c_cam_position[cam_idx]);
            result = result - C1 * y * coefficients_ptr[0]
                            + C1 * z * coefficients_ptr[1]
                            - C1 * x * coefficients_ptr[2];
            if (active_sh_bases > 4) {
                const float xx = x * x, yy = y * y, zz = z * z;
                const float xy = x * y, xz = x * z, yz = y * z;
                result = result + C2a * xy * coefficients_ptr[3]
                                - C2a * yz * coefficients_ptr[4]
                                + (C2b * zz - C2c) * coefficients_ptr[5]
                                - C2a * xz * coefficients_ptr[6]
                                + C2d * (xx - yy) * coefficients_ptr[7];
                if (active_sh_bases > 9) {
                    result = result + y * (C3a * yy - C3b * xx) * coefficients_ptr[8]
                                    + C3c * xy * z * coefficients_ptr[9]
                                    + y * (C3d - C3e * zz) * coefficients_ptr[10]
                                    + z * (C3f * zz - C3g) * coefficients_ptr[11]
                                    + x * (C3d - C3e * zz) * coefficients_ptr[12]
                                    + C3h * z * (xx - yy) * coefficients_ptr[13]
                                    + x * (C3b * yy - C3a * xx) * coefficients_ptr[14];
                }
            }
        }
        return result;
    }
}
