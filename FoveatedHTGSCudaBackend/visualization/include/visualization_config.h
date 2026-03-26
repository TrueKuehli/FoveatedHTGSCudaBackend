#pragma once

#include "config.h"
#define DEF inline constexpr


namespace htgs_foveated::visualization::config {
    // debugging constants
    DEF bool debug_visualization = false;

    // visualization constants
    DEF int gaze_visualization_width = 45;
    DEF int gaze_visualization_size = gaze_visualization_width * gaze_visualization_width;
    DEF bool gaze_visualization_circular = true;
    DEF int num_border_pixels_small = rasterization::config::tile_width_small * 2 + rasterization::config::tile_height_small * 2 - 4;
    DEF int num_border_pixels_large = rasterization::config::tile_width_large * 2 + rasterization::config::tile_height_large * 2 - 4;
}

#undef DEF
