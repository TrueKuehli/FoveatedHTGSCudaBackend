#pragma once

#include "utils/kernel_utils.cuh"
#include "utils/sh_utils.cuh"


namespace htgs_foveated::rasterization::kernels::background_model {
    template<uint8_t cam_idx>
    __forceinline__ __device__ float3 eval_sh_background_model(const float pixel_x, const float pixel_y) {
        // computation adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L340
        const float4 pixel_coords = make_float4(pixel_x, pixel_y, 1.0, 1.0);
        const float3 pixel_coords_transformed = normalize(make_float3(
            dot(c_VPR_inv[cam_idx][0], pixel_coords),
            dot(c_VPR_inv[cam_idx][1], pixel_coords),
            dot(c_VPR_inv[cam_idx][2], pixel_coords)
        ));

        // Early exit for pixels >10% below the horizon
        if (pixel_coords_transformed.y > 0.1f) return make_float3(0.0f, 0.0f, 0.0f);

        const auto [x, y, z] = normalize(pixel_coords_transformed);
        const float xx = x * x, yy = y * y, zz = z * z;
        const float xy = x * y, xz = x * z, yz = y * z;
        const float3 sh_eval = 0.5f + C0 * c_background_sh_coeff[0]
                             - C1 * y * c_background_sh_coeff[1]
                             + C1 * z * c_background_sh_coeff[2]
                             - C1 * x * c_background_sh_coeff[3]
                             + C2a * xy * c_background_sh_coeff[4]
                             - C2a * yz * c_background_sh_coeff[5]
                             + (C2b * zz - C2c) * c_background_sh_coeff[6]
                             - C2a * xz * c_background_sh_coeff[7]
                             + C2d * (xx - yy) * c_background_sh_coeff[8]
                             + y * (C3a * yy - C3b * xx) * c_background_sh_coeff[9]
                             + C3c * xy * z * c_background_sh_coeff[10]
                             + y * (C3d - C3e * zz) * c_background_sh_coeff[11]
                             + z * (C3f * zz - C3g) * c_background_sh_coeff[12]
                             + x * (C3d - C3e * zz) * c_background_sh_coeff[13]
                             + C3h * z * (xx - yy) * c_background_sh_coeff[14]
                             + x * (C3b * yy - C3a * xx) * c_background_sh_coeff[15];
        const float3 color = make_float3(
            __saturatef(tanhf(sh_eval.x) * 0.5f + 0.5f),
            __saturatef(tanhf(sh_eval.y) * 0.5f + 0.5f),
            __saturatef(tanhf(sh_eval.z) * 0.5f + 0.5f)
        );

        // Interpolate to black below the horizon
        return pixel_coords_transformed.y > 0.0f ? (1.0f - pixel_coords_transformed.y / 0.1f) * color : color;
    }


    template<int width, int height, uint8_t cam_idx>
    __forceinline__ __device__ float3 eval_tex_background_model(const float pixel_x, const float pixel_y, const float* texture_data) {
        const float4 pixel_coords = make_float4(pixel_x, pixel_y, 1.0, 1.0);
        const float3 pixel_coords_transformed = normalize(make_float3(
            dot(c_VPR_inv[cam_idx][0], pixel_coords),
            dot(c_VPR_inv[cam_idx][1], pixel_coords),
            dot(c_VPR_inv[cam_idx][2], pixel_coords)
        ));

        // Early exit for pixels >10% below the horizon
        if (pixel_coords_transformed.y > 0.1f) return make_float3(0.0f, 0.0f, 0.0f);

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

        // Interpolate to black below the horizon
        return pixel_coords_transformed.y > 0.0f ? (1.0f - pixel_coords_transformed.y / 0.1f) * color : color;
    }
}
