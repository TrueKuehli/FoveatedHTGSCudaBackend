#pragma once

#include "helper_math.h"
#include <cstdint>

namespace htgs::rasterization::shared_kernels {

    template <typename KeyT>
    __global__ void create_instances_cu(
        const uint* primitive_n_touched_tiles,
        const uint* primitive_offsets,
        const uint4* primitive_screen_bounds,
        KeyT* instance_keys,
        uint* instance_primitive_indices,
        const uint* render_mask,
        const uint2 gaze_position_tiles,
        const uint grid_width,
        const uint n_primitives,
        const uint foveation_radius_tiles);

    __global__ void create_instances_cu(
        const uint* primitive_n_touched_tiles,
        const uint* primitive_offsets,
        const uint4* primitive_screen_bounds,
        const float* primitive_depths,
        uint64_t* instance_keys,
        uint* instance_primitive_indices,
        const uint* render_mask,
        const uint2 gaze_position_tiles,
        const uint grid_width,
        const uint n_primitives,
        const uint foveation_radius_tiles);

    template <typename KeyT>
    __global__ void extract_instance_ranges_cu(
        const KeyT* instance_keys,
        uint2* tile_instance_ranges,
        const uint n_instances);
    
    __global__ void extract_instance_ranges_cu(
        const uint64_t* instance_keys,
        uint2* tile_instance_ranges,
        const uint n_instances);


    __global__ void fill_tile_index_num_tiles(
        uint* tile_index_map_num_tiles,
        const uint* tile_mask,
        const uint2 gaze_position_tiles,
        const uint num_tiles_total,
        const uint grid_width,
        const uint foveation_radius,
        const uint num_small_tiles
    );

    __global__ void build_tile_index_map(
        uint* tile_index_map,
        const uint* tile_index_map_num_tiles,
        const uint* tile_index_map_offsets,
        const uint num_tiles_total,
        const uint num_small_tiles
    );
}
