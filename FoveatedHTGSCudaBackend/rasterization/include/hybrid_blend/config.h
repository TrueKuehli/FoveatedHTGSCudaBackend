#pragma once

#include "helper_math.h"

#define DEF inline constexpr

namespace htgs::rasterization::hybrid_blend::config {
    // debugging constants
    DEF bool debug_inference = false;
    DEF bool debug_fast_inference = false;
    // rendering constants
    DEF float transmittance_threshold = 1e-4f;
    DEF float max_fragment_alpha = 1.0f; // 3dgs uses 0.99f
    DEF float min_alpha_threshold_rcp = 255.0f;
    DEF float min_alpha_threshold = 1.0f / min_alpha_threshold_rcp; // 0.00392156862
    DEF float max_cutoff_sq = 11.0825270903f; // logf(min_alpha_threshold_rcp * min_alpha_threshold_rcp)
    DEF float min_alpha_threshold_core_rcp = 20.0f;
    DEF float min_alpha_threshold_core = 1.0f / min_alpha_threshold_core_rcp; // 0.05

    DEF int tile_width_small = 8;
    DEF int tile_height_small = 8;
    DEF int tile_width_large = 16;
    DEF int tile_height_large = 16;
    DEF int tile_stride_x = tile_width_large / tile_width_small;
    DEF int tile_stride_y = tile_height_large / tile_height_small;
    DEF int num_small_tiles_per_large_tile = tile_stride_x * tile_stride_y;

    DEF int foveation_radius = 375; // in pixels
    DEF int foveation_radius_tiles = (foveation_radius + tile_width_large - 1) / tile_width_large; // in tiles
    DEF int blend_region = 125;     // in pixels

    // block size constants
    DEF int block_size_create_tile_index_map = 256;
    DEF int block_size_preprocess = 256;
    DEF int block_size_create_instances = 256;
    DEF int block_size_extract_instance_ranges = 256;
    DEF int block_size_blend = tile_width_small * tile_height_small;

    DEF int gaze_visualization_width = 13;
    DEF int gaze_visualization_size = gaze_visualization_width * gaze_visualization_width;
    DEF bool gaze_visualization_circular = true;
}

namespace config = htgs::rasterization::hybrid_blend::config;

#undef DEF
