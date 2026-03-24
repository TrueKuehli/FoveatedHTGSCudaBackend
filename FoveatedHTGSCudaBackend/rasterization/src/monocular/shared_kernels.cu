#include "helper_math.h"
#include "kernels/monocular/shared_kernels.cuh"
#include "utils/monocular/kernel_utils.cuh"
#include <cstdint>


namespace htgs_foveated::rasterization::kernels::monocular::shared {
    template <typename KeyT>
    __global__ void extract_instance_ranges_cu(
        const KeyT* instance_keys,
        uint2* tile_instance_ranges,
        const uint n_instances)
    {
        const uint instance_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (instance_idx >= n_instances) return;
        const KeyT instance_tile_idx = instance_keys[instance_idx];
        if (instance_tile_idx == ~static_cast<KeyT>(0)) return;  // Reject non-set keys
        if (instance_idx == 0) tile_instance_ranges[instance_tile_idx].x = 0;
        else {
            const KeyT previous_instance_tile_idx = instance_keys[instance_idx - 1];
            if (instance_tile_idx != previous_instance_tile_idx) {
                tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;
                tile_instance_ranges[instance_tile_idx].x = instance_idx;
            }
        }
        if (instance_idx == n_instances - 1) tile_instance_ranges[instance_tile_idx].y = n_instances;
    }

    __global__ void extract_instance_ranges_cu(
        const uint64_t* instance_keys,
        uint2* tile_instance_ranges,
        const uint n_instances)
    {
        const uint instance_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (instance_idx >= n_instances) return;
        const uint64_t instance_key = instance_keys[instance_idx];
        if (instance_key == ~static_cast<uint64_t>(0)) return;  // Reject non-set keys
        const uint instance_tile_idx = instance_key >> 32;
        if (instance_idx == 0) tile_instance_ranges[instance_tile_idx].x = 0;
        else {
            const uint64_t previous_instance_key = instance_keys[instance_idx - 1];
            const uint previous_instance_tile_idx = previous_instance_key >> 32;
            if (instance_tile_idx != previous_instance_tile_idx) {
                tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;
                tile_instance_ranges[instance_tile_idx].x = instance_idx;
            }
        }
        if (instance_idx == n_instances - 1) tile_instance_ranges[instance_tile_idx].y = n_instances;
    }

    template __global__ void extract_instance_ranges_cu<uint>(
        const uint*, uint2*, const uint);
    template __global__ void extract_instance_ranges_cu<ushort>(
        const ushort*, uint2*, const uint);

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
