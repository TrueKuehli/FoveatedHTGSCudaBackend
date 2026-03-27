#pragma once

#include "helper_math.h"
#include "utils/enums.h"
#include "utils/stereo/kernel_utils.cuh"
#include <cstdint>


namespace htgs_foveated::rasterization::kernels::stereo::shared {

    template <typename KeyT, uint foveation_radius_tiles, uint num_small_tiles, uint8_t cam_idx>
    __global__ inline void create_instances_cu(
        const uint* primitive_n_touched_tiles,
        const uint* primitive_offsets,
        const uint4* primitive_screen_bounds,
        KeyT* instance_keys,
        uint* instance_primitive_indices,
        const float2 gaze_position_tiles,
        const uint grid_width,
        const uint n_primitives)
    {
        const uint primitive_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0) return;
        const uint4 screen_bounds = primitive_screen_bounds[primitive_idx];
        uint offset = (primitive_idx == 0) ? 0 : primitive_offsets[primitive_idx - 1];
        for (uint y = screen_bounds.z; y < screen_bounds.w; ++y) {
            for (uint x = screen_bounds.x; x < screen_bounds.y; ++x) {
                const KeyT tile_idx = y * grid_width + x;
                const int mask_byte_idx = tile_idx / 32;
                const int mask_bit_idx = tile_idx % 32;
                if ((c_render_mask[cam_idx][mask_byte_idx] & (1 << mask_bit_idx)) == 0) continue;

                if (is_in_fovea<foveation_radius_tiles>(make_int2(static_cast<int>(x), static_cast<int>(y)), grid_width, gaze_position_tiles)) {
                    // Tile is in fovea, so create instances for each small tile
                    #pragma unroll num_small_tiles
                    for (uint i = 0; i < num_small_tiles; ++i) {
                        instance_keys[offset] = tile_idx * num_small_tiles + i;
                        instance_primitive_indices[offset] = primitive_idx;
                        offset++;
                    }
                } else {
                    instance_keys[offset] = tile_idx * num_small_tiles;
                    instance_primitive_indices[offset] = primitive_idx;
                    offset++;
                }
            }
        }
    }

    template <uint foveation_radius_tiles, uint num_small_tiles, uint8_t cam_idx>
    __global__ inline void create_instances_cu(
        const uint* primitive_n_touched_tiles,
        const uint* primitive_offsets,
        const uint4* primitive_screen_bounds,
        const float* primitive_depths,
        uint64_t* instance_keys,
        uint* instance_primitive_indices,
        const float2 gaze_position_tiles,
        const uint grid_width,
        const uint n_primitives)
    {
        const uint primitive_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0) return;
        const uint4 screen_bounds = primitive_screen_bounds[primitive_idx];
        uint offset = (primitive_idx == 0) ? 0 : primitive_offsets[primitive_idx - 1];
        const uint64_t depth_key = __float_as_uint(primitive_depths[primitive_idx]);
        for (uint y = screen_bounds.z; y < screen_bounds.w; ++y) {
            for (uint x = screen_bounds.x; x < screen_bounds.y; ++x) {
                const uint64_t tile_idx = y * grid_width + x;
                const int mask_byte_idx = tile_idx / 32;
                const int mask_bit_idx = tile_idx % 32;
                if ((c_render_mask[cam_idx][mask_byte_idx] & (1 << mask_bit_idx)) == 0) continue;

                if (is_in_fovea<foveation_radius_tiles>(make_int2(static_cast<int>(x), static_cast<int>(y)), grid_width, gaze_position_tiles)) {
                    // Tile is in fovea, so create instances for each small tile
                    #pragma unroll num_small_tiles
                    for (uint i = 0; i < num_small_tiles; ++i) {
                        instance_keys[offset] = ((tile_idx * num_small_tiles + i) << 32) | depth_key;
                        instance_primitive_indices[offset] = primitive_idx;
                        offset++;
                    }
                } else {
                    instance_keys[offset] = ((tile_idx * num_small_tiles) << 32) | depth_key;
                    instance_primitive_indices[offset] = primitive_idx;
                    offset++;
                }
            }
        }
    }

    template <typename KeyT>
    __global__ void extract_instance_ranges_cu(
        const KeyT* instance_keys,
        uint2* tile_instance_ranges,
        const uint n_instances);

    __global__ void extract_instance_ranges_cu(
        const uint64_t* instance_keys,
        uint2* tile_instance_ranges,
        const uint n_instances);


    template <int foveation_radius_tiles, int num_small_tiles, uint8_t cam_idx>
    __global__ inline void fill_tile_index_num_tiles(
        uint* tile_index_map_num_tiles,
        const float2 gaze_position_tiles,
        const uint num_tiles_total,
        const uint grid_width
    ) {
        const uint tile_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (tile_idx >= num_tiles_total) return;

        const uint mask_byte_idx = tile_idx / 32;
        const uint mask_bit_idx = tile_idx % 32;

        if ((c_render_mask[cam_idx][mask_byte_idx] & (1 << mask_bit_idx)) != 0) {
            if (is_in_fovea<foveation_radius_tiles>(tile_idx, grid_width, gaze_position_tiles)) {
                // Tiles in the fovea get split into n small tiles
                tile_index_map_num_tiles[tile_idx] = num_small_tiles;
            } else {
                tile_index_map_num_tiles[tile_idx] = 1;
            }
        } else {
            tile_index_map_num_tiles[tile_idx] = 0;
        }
    }

    template <int num_small_tiles, int blend_radius_tiles>
    __global__ inline void build_tile_index_map(
        uint* tile_index_map,
        TileType* tile_type_map,
        const uint* tile_index_map_num_tiles,
        const uint* tile_index_map_offsets,
        const float2 gaze_position_tiles,
        const uint grid_width,
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
            tile_type_map[base_idx] = TileType::PERIPHERY;
        } else {
            const TileType tile_type =
                    is_in_fovea<blend_radius_tiles>(tile_idx, grid_width, gaze_position_tiles) ?
                    TileType::FOVEA :
                    TileType::BLENDED;

            // Tile gets split
            #pragma unroll
            for (uint i = 0; i < num_small_tiles; ++i) {
                tile_index_map[base_idx + i] = tile_idx * num_small_tiles + i;
                tile_type_map[base_idx + i] = tile_type;
            }
        }
    }

    __global__ void get_partition_ranges_cu(
        uint2* partition_ranges,
        const TileType* tile_type_map,
        const uint n_tiles
    );
}
