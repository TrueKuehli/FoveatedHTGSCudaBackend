#pragma once

#include "helper_math.h"
#include "utils.h"
#include "utils/enums.h"
#include "utils/kernel_utils.cuh"
#include <cstdint>
#include <limits>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;


namespace htgs_foveated::rasterization::kernels::shared {

    template <typename KeyT, uint foveation_radius_tiles, uint num_small_tiles, uint8_t cam_idx>
    __global__ void create_instances_cu(
        const uint* primitive_n_touched_tiles,
        const uint* primitive_offsets,
        const ushort4* primitive_screen_bounds,
        KeyT* instance_keys,
        uint* instance_primitive_indices,
        const uint32_t* visibility_mask,
        const float2 gaze_position_tiles,
        const uint grid_width,
        const uint n_primitives)
    {
        constexpr uint warp_size = 32;
        constexpr uint n_sequential_threshold = 2;
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<warp_size>(block);
        uint primitive_idx = cg::this_grid().thread_rank();
        const uint thread_rank = block.thread_rank();
        const uint warp_idx = warp.meta_group_rank();
        const uint warp_start = warp_idx * warp_size;
        const uint lane_idx = warp.thread_rank();
        const uint previous_lanes_mask = (1 << lane_idx) - 1;

        bool active = true;
        if (primitive_idx >= n_primitives) {
            active = false;
            primitive_idx = n_primitives - 1;
        }

        const uint tile_count_init = primitive_n_touched_tiles[primitive_idx];
        if (tile_count_init == 0) active = false;

        if (warp.ballot(active) == 0) return;

        const ushort4 screen_bounds = primitive_screen_bounds[primitive_idx];
        const uint screen_bounds_width = static_cast<uint>(screen_bounds.y - screen_bounds.x);
        const uint instance_count = static_cast<uint>(screen_bounds.w - screen_bounds.z) * screen_bounds_width;

        uint current_write_offset = (primitive_idx == 0) ? 0 : primitive_offsets[primitive_idx - 1];

        for (uint instance_idx = 0; active && instance_idx < instance_count && instance_idx < n_sequential_threshold; instance_idx++) {
            const uint tile_x = screen_bounds.x + (instance_idx % screen_bounds_width);
            const uint tile_y = screen_bounds.z + (instance_idx / screen_bounds_width);
            const uint tile_idx = tile_y * grid_width + tile_x;

            if ((visibility_mask[tile_idx / 32u] & (1 << (tile_idx % 32u))) == 0) continue;

            const KeyT instance_key = static_cast<KeyT>(tile_idx * num_small_tiles);
            if (is_in_fovea<foveation_radius_tiles>(tile_x, tile_y, grid_width, gaze_position_tiles)) {
                // Tile is in fovea, so create instances for each small tile
                #pragma unroll num_small_tiles
                for (ushort i = 0; i < num_small_tiles; ++i) {
                    instance_keys[current_write_offset] = instance_key + i;
                    instance_primitive_indices[current_write_offset] = primitive_idx;
                    current_write_offset++;
                }
            } else {
                instance_keys[current_write_offset] = instance_key;
                instance_primitive_indices[current_write_offset] = primitive_idx;
                current_write_offset++;
            }
        }

        const bool compute_cooperatively = active && instance_count > n_sequential_threshold;
        const uint remaining_threads = warp.ballot(compute_cooperatively);
        if (remaining_threads == 0) return;

        __shared__ ushort4 collected_screen_bounds[config::block_size_create_instances];
        collected_screen_bounds[thread_rank] = screen_bounds;

        const uint n_remaining_threads = __popc(remaining_threads);
        for (uint n = 0; n < n_remaining_threads && n < warp_size; n++) {
            const uint current_lane = __fns(remaining_threads, 0, n + 1);
            const uint primitive_idx_coop = warp.shfl(primitive_idx, current_lane);
            uint current_write_offset_coop = warp.shfl(current_write_offset, current_lane);

            const uint read_offset_shared = warp_start + current_lane;
            const ushort4 screen_bounds_coop = collected_screen_bounds[read_offset_shared];

            const uint screen_bounds_width_coop = static_cast<uint>(screen_bounds_coop.y - screen_bounds_coop.x);
            const uint instance_count_coop = screen_bounds_width_coop * static_cast<uint>(screen_bounds_coop.w - screen_bounds_coop.z);

            const uint remaining_instance_count = instance_count_coop - n_sequential_threshold;
            const uint n_iterations = div_round_up(remaining_instance_count, warp_size);
            for (uint i = 0; i < n_iterations; i++) {
                const uint instance_idx = i * warp_size + lane_idx + n_sequential_threshold;
                const uint tile_x = screen_bounds_coop.x + (instance_idx % screen_bounds_width_coop);
                const uint tile_y = screen_bounds_coop.z + (instance_idx / screen_bounds_width_coop);
                const uint tile_idx = tile_y * grid_width + tile_x;
                const bool valid_tile = (visibility_mask[tile_idx / 32u] & (1 << (tile_idx % 32u))) != 0;
                const bool write = instance_idx < instance_count_coop && valid_tile;
                const uint write_ballot = warp.ballot(write);
                const bool is_fovea = write && is_in_fovea<foveation_radius_tiles>(tile_x, tile_y, grid_width, gaze_position_tiles);
                const uint is_fovea_ballot = warp.ballot(is_fovea);
                if (write) {
                    uint write_offset = current_write_offset_coop + __popc(write_ballot & previous_lanes_mask) + (num_small_tiles - 1) * __popc(is_fovea_ballot & previous_lanes_mask);
                    const KeyT instance_key = static_cast<KeyT>(tile_idx * num_small_tiles);
                    if (is_fovea) {
                        // Tile is in fovea, so create instances for each small tile
                        #pragma unroll num_small_tiles
                        for (ushort i = 0; i < num_small_tiles; ++i) {
                            instance_keys[write_offset] = instance_key + i;
                            instance_primitive_indices[write_offset] = primitive_idx_coop;
                            write_offset++;
                        }
                    } else {
                        instance_keys[write_offset] = instance_key;
                        instance_primitive_indices[write_offset] = primitive_idx_coop;
                    }
                }
                const uint n_written = __popc(write_ballot) + (num_small_tiles - 1) * __popc(is_fovea_ballot);
                current_write_offset_coop += n_written;
            }
            warp.sync();
        }
    }


    template <typename KeyT>
    __global__ void extract_instance_ranges_cu(
        const KeyT* instance_keys,
        uint2* tile_instance_ranges,
        const uint n_instances)
    {
        const uint instance_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (instance_idx >= n_instances) return;
        const KeyT instance_tile_idx = instance_keys[instance_idx];
        if (instance_idx == 0) tile_instance_ranges[instance_tile_idx].x = 0;
        else {
            const KeyT previous_instance_tile_idx = instance_keys[instance_idx - 1];
            if (instance_tile_idx != previous_instance_tile_idx) {
                tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;
                // Don't set the start of the next range for sentinel keys (which are always at the end, so no check necessary for previous_instance_tile_idx)
                if (instance_tile_idx != std::numeric_limits<KeyT>::max()) tile_instance_ranges[instance_tile_idx].x = instance_idx;
            }
        }
        if (instance_idx == n_instances - 1 && instance_tile_idx != std::numeric_limits<KeyT>::max()) tile_instance_ranges[instance_tile_idx].y = n_instances;
    }


    template <int foveation_radius_tiles, int num_small_tiles, uint8_t cam_idx>
    __global__ void fill_tile_index_num_tiles(
        uint* tile_index_map_num_tiles,
        const uint32_t* visibility_mask,
        const float2 gaze_position_tiles,
        const uint num_tiles_total,
        const uint grid_width
    ) {
        const uint tile_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (tile_idx >= num_tiles_total) return;

        const uint mask_byte_idx = tile_idx / 32u;
        const uint mask_bit_idx = tile_idx % 32u;

        if ((visibility_mask[mask_byte_idx] & (1 << mask_bit_idx)) != 0) {
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
    __global__ void build_tile_index_map(
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
    ) {
        const uint tile_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (tile_idx >= n_tiles) return;
        if (tile_idx == 0) return;

        TileType current_type = tile_type_map[tile_idx];
        TileType previous_type = tile_type_map[tile_idx - 1];
        if (tile_idx != 0 && current_type != previous_type) {
            partition_ranges[static_cast<uint8_t>(current_type)].x = tile_idx;
            partition_ranges[static_cast<uint8_t>(previous_type)].y = tile_idx;
        }
        if (tile_idx == n_tiles - 1) partition_ranges[static_cast<uint8_t>(current_type)].y = n_tiles;
    }

}
