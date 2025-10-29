#pragma once

#include "helper_math.h"
#include "kernel_utils.cuh"
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
        const uint foveation_radius_tiles,
        const uint num_small_tiles);

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
        const uint foveation_radius_tiles,
        const uint num_small_tiles);

    template <typename KeyT>
    __global__ void extract_instance_ranges_cu(
        const KeyT* instance_keys,
        uint2* tile_instance_ranges,
        const uint n_instances);

    __global__ void extract_instance_ranges_cu(
        const uint64_t* instance_keys,
        uint2* tile_instance_ranges,
        const uint n_instances);


    template <int foveation_radius_tiles, int num_small_tiles>
    __global__ inline void fill_tile_index_num_tiles(
        uint* tile_index_map_num_tiles,
        const uint* tile_mask,
        const uint2 gaze_position_tiles,
        const uint num_tiles_total,
        const uint grid_width
    ) {
        const uint tile_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (tile_idx >= num_tiles_total) return;

        const uint mask_byte_idx = tile_idx / 32;
        const uint mask_bit_idx = tile_idx % 32;

        if ((tile_mask[mask_byte_idx] & (1 << mask_bit_idx)) != 0) {
            if (is_in_fovea(tile_idx, grid_width, gaze_position_tiles, foveation_radius_tiles)) {
                // Tiles in the fovea get split into n small tiles
                tile_index_map_num_tiles[tile_idx] = num_small_tiles;
            } else {
                tile_index_map_num_tiles[tile_idx] = 1;
            }
        } else {
            tile_index_map_num_tiles[tile_idx] = 0;
        }
    }

    template <int num_small_tiles>
    __global__ inline void build_tile_index_map(
        uint* tile_index_map,
        const uint* tile_index_map_num_tiles,
        const uint* tile_index_map_offsets,
        const uint num_tiles_total
    ) {
        // The tile index map essentially identifies the (location of) each tile in the tile grid, knowing only the index among all active tiles
        // To allow tiles to be split into smaller tiles for foveated rendering, the resulting tile index corresponds to:
        // - tile_idx / num_small_tiles == the large tile index
        // - tile_idx % num_small_tiles == the small tile index within the large tile, in the order top-left, top-right, bottom-left, bottom-right

        const uint tile_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (tile_idx >= num_tiles_total) return;

        if (tile_index_map_num_tiles[tile_idx] == 0) {
            return;
        }

        const uint base_idx = tile_idx == 0 ? 0 : tile_index_map_offsets[tile_idx - 1];
        if (tile_index_map_num_tiles[tile_idx] == 1) {
            tile_index_map[base_idx] = tile_idx * num_small_tiles;
        } else {
            // Tile gets split
            #pragma unroll
            for (uint i = 0; i < num_small_tiles; ++i) {
                tile_index_map[base_idx + i] = tile_idx * num_small_tiles + i;
            }
        }
    }
}
