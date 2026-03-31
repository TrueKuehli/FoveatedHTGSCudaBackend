#pragma once


#include "enums.h"

namespace htgs_foveated::visualization {

    void clear_image(
        float* image,
        const int width,
        const int height);

    void visualize_gaze_position(
        float* image,
        const int width,
        const int height,
        const int visualization_size,
        const float2* gaze_position,
        const float3 gaze_color,
        const GazeVisualizationType visualization_type,
        const bool to_chw);

   void visualize_tile_boundaries(
        std::function<char* (size_t)> per_tile_buffers_func,
        std::function<char* (size_t)> per_subtile_buffers_func,
        float* image,
        const float2* gaze_position,
        const uint* visibility_mask,
        const int width,
        const int height,
        const bool to_chw);

}
