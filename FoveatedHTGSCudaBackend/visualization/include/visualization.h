#pragma once

namespace htgs_foveated::visualization {

    void visualize_gaze_position(
        float* image,
        const int width,
        const int height,
        const float2* gaze_position,
        const bool to_chw);

   void visualize_tile_boundaries(
        std::function<char* (size_t)> per_tile_buffers_func,
        std::function<char* (size_t)> per_subtile_buffers_func,
        float* image,
        const float2* gaze_position,
        const uint* render_mask,
        const int width,
        const int height,
        const bool to_chw);

}
