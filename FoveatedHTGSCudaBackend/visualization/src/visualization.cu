#include "config.h"
#include "helper_math.h"
#include "kernels/monocular/shared_kernels.cuh"
#include "kernels/visualization.cuh"
#include "utils/buffer_utils.h"
#include "utils/enums.h"
#include "enums.h"
#include "utils.h"
#include "visualization.h"
#include "visualization_config.h"
#include <cub/cub.cuh>
#include <functional>
#include <variant>
#include <utility>
#include <type_traits>


void htgs_foveated::visualization::clear_image(
    float* image,
    const int width,
    const int height
) {
    cudaMemset(image, 0, sizeof(float) * width * height * 3);
}


void htgs_foveated::visualization::visualize_gaze_position(
    float* image,
    const int width,
    const int height,
    const int visualization_size,
    const float2* gaze_position,
    const float3 gaze_color,
    const GazeVisualizationType visualization_type,
    const bool to_chw
) {
    // If gaze_position is null, assume we've already written it e.g. during rasterization to the respective constant memory location
    if (gaze_position != nullptr) {
        const float2 gaze_position_clamped = make_float2(
            clamp(gaze_position->x, 0.0f, static_cast<float>(width - 1)),
            clamp(gaze_position->y, 0.0f, static_cast<float>(height - 1))
        );
        cudaMemcpyToSymbol(c_gaze_position_cuda, &gaze_position_clamped, sizeof(float2), 0, cudaMemcpyHostToDevice);
    }

    // Draw a dot at the gaze position for visualization
    const dim3 dot_grid(div_round_up(visualization_size, rasterization::config::tile_width_small),
                        div_round_up(visualization_size, rasterization::config::tile_width_small), 1);
    const dim3 dot_block(rasterization::config::tile_width_small, rasterization::config::tile_width_small, 1);
    kernels::visualize_gaze_cu<<<dot_grid, dot_block>>>(
        image,
        width,
        height,
        gaze_color,
        visualization_size,
        visualization_type,
        to_chw
    );
}


void htgs_foveated::visualization::visualize_tile_boundaries(
    std::function<char* (size_t)> per_tile_buffers_func,
    std::function<char* (size_t)> per_subtile_buffers_func,
    float* image,
    const float2* gaze_position,
    const uint* render_mask,
    const int width,
    const int height,
    const bool to_chw)
{
    const float2 gaze_position_clamped = make_float2(
        clamp(gaze_position->x, 0.0f, static_cast<float>(width - 1)),
        clamp(gaze_position->y, 0.0f, static_cast<float>(height - 1))
    );

    const dim3 grid_large(div_round_up(width, rasterization::config::tile_width_large),
                          div_round_up(height, rasterization::config::tile_height_large), 1);
    const dim3 grid(grid_large.x * rasterization::config::tile_stride_x,
                    grid_large.y * rasterization::config::tile_stride_y, 1);
    const dim3 block(visualization::config::num_border_pixels_small, 1);
    const dim3 block_large(visualization::config::num_border_pixels_large, 1);
    const int n_tiles_large = grid_large.x * grid_large.y;
    const int n_tiles = grid.x * grid.y;
    cudaMemcpyToSymbol(c_render_mask, render_mask, div_round_up(grid_large.x * grid_large.y, 8U), 0, cudaMemcpyDeviceToDevice);

    // Round gaze to nearest large tile (top left corner of tile)
    const float2 gaze_position_tiles = make_float2(
        static_cast<float>(max(0, min(static_cast<int>(grid.x - 1),
                (static_cast<int>(gaze_position_clamped.x) + rasterization::config::tile_width_large / 2)
                / rasterization::config::tile_width_large))),
        static_cast<float>(max(0, min(static_cast<int>(grid.y - 1),
                (static_cast<int>(gaze_position_clamped.y) + rasterization::config::tile_width_large / 2)
                / rasterization::config::tile_width_large)))
    );

    char* per_tile_buffers_blob = per_tile_buffers_func(rasterization::required<rasterization::PerTileBuffers>(n_tiles_large));
    rasterization::PerTileBuffers per_tile_buffers = rasterization::PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles_large);
    char* per_sub_tile_buffers_blob = per_subtile_buffers_func(rasterization::required<rasterization::PerSubTileBuffers>(n_tiles));
    rasterization::PerSubTileBuffers per_sub_tile_buffers = rasterization::PerSubTileBuffers::from_blob(per_sub_tile_buffers_blob, n_tiles);

    static cudaStream_t memset_stream = 0;
    if constexpr (!visualization::config::debug_visualization) {
        static bool memset_stream_initialized = false;
        if (!memset_stream_initialized) {
            cudaStreamCreate(&memset_stream);
            memset_stream_initialized = true;
        }
        cudaMemsetAsync(per_sub_tile_buffers.partition_ranges, 0, sizeof(rasterization::PartitionRanges), memset_stream);
    }
    else cudaMemset(per_sub_tile_buffers.partition_ranges, 0, sizeof(rasterization::PartitionRanges));

    // Build tile index map (so we only need to process [0, num_active_tiles), which we can map back to the "true" tile index)
    htgs_foveated::rasterization::kernels::monocular::shared::fill_tile_index_num_tiles
            <rasterization::config::foveation_radius_tiles, rasterization::config::num_small_tiles_per_large_tile>
            <<<div_round_up(n_tiles_large, rasterization::config::block_size_create_tile_index_map),
                            rasterization::config::block_size_create_tile_index_map>>>
    (
        per_tile_buffers.tile_index_map_num_tiles,
        gaze_position_tiles,
        n_tiles_large,
        grid_large.x
    );
    CHECK_CUDA(visualization::config::debug_visualization, "fill_tile_index_num_tiles")
    cub::DeviceScan::InclusiveSum(
        per_tile_buffers.cub_workspace, per_tile_buffers.cub_workspace_size,
        per_tile_buffers.tile_index_map_num_tiles, per_tile_buffers.tile_index_map_offsets,
        n_tiles_large
    );
    CHECK_CUDA(visualization::config::debug_visualization, "cub::DeviceScan::InclusiveSum (index_map)")
    htgs_foveated::rasterization::kernels::monocular::shared::build_tile_index_map
            <rasterization::config::num_small_tiles_per_large_tile, rasterization::config::blend_radius_tiles>
            <<<div_round_up(n_tiles_large, rasterization::config::block_size_create_tile_index_map),
                            rasterization::config::block_size_create_tile_index_map>>>
    (
        per_sub_tile_buffers.tile_index_map,
        per_sub_tile_buffers.tile_type,
        per_tile_buffers.tile_index_map_num_tiles,
        per_tile_buffers.tile_index_map_offsets,
        gaze_position_tiles,
        grid_large.x,
        n_tiles_large
    );
    CHECK_CUDA(visualization::config::debug_visualization, "build_tile_index_map")

    uint num_active_tiles;
    cudaMemcpy(&num_active_tiles, per_tile_buffers.tile_index_map_offsets + n_tiles_large - 1, sizeof(uint), cudaMemcpyDeviceToHost);
    CHECK_CUDA(visualization::config::debug_visualization, "Fetch num_active_tiles")

    cub::DeviceRadixSort::SortPairs(
        per_sub_tile_buffers.cub_workspace, per_sub_tile_buffers.cub_workspace_size,
        reinterpret_cast<uint8_t*>(per_sub_tile_buffers.tile_type), reinterpret_cast<uint8_t*>(per_sub_tile_buffers.tile_type_partitioned),
        per_sub_tile_buffers.tile_index_map, per_sub_tile_buffers.tile_index_map_partitioned,
        num_active_tiles, 0, rasterization::NUM_TILE_TYPE_BITS
    );
    CHECK_CUDA(visualization::config::debug_visualization, "Sort tiles by type")

    if constexpr (!visualization::config::debug_visualization) cudaStreamSynchronize(memset_stream);
    htgs_foveated::rasterization::kernels::monocular::shared::get_partition_ranges_cu
            <<<div_round_up(static_cast<int>(num_active_tiles), rasterization::config::block_size_get_partition_ranges),
                            rasterization::config::block_size_get_partition_ranges>>>
    (
        reinterpret_cast<uint2*>(per_sub_tile_buffers.partition_ranges),
        per_sub_tile_buffers.tile_type_partitioned,
        num_active_tiles
    );
    CHECK_CUDA(visualization::config::debug_visualization, "Partition tiles by type")

    rasterization::PartitionRanges partition_ranges_cpu;
    cudaMemcpy(&partition_ranges_cpu, per_sub_tile_buffers.partition_ranges, sizeof(rasterization::PartitionRanges), cudaMemcpyDeviceToHost);
    const int num_tiles_fovea = partition_ranges_cpu.fovea_tiles_range.y - partition_ranges_cpu.fovea_tiles_range.x;
    const int num_tiles_periphery = partition_ranges_cpu.periphery_tiles_range.y - partition_ranges_cpu.periphery_tiles_range.x;
    const int num_tiles_blended = partition_ranges_cpu.blended_tiles_range.y - partition_ranges_cpu.blended_tiles_range.x;

    static cudaStream_t visualize_fovea_stream = 0;
    static cudaStream_t visualize_periphery_stream = 0;
    static cudaStream_t visualize_blended_tiles_stream = 0;
    static bool visualize_streams_initialized = false;
    if (!visualize_streams_initialized) {
        cudaStreamCreate(&visualize_fovea_stream);
        cudaStreamCreate(&visualize_periphery_stream);
        cudaStreamCreate(&visualize_blended_tiles_stream);
        visualize_streams_initialized = true;
    }

    const dim3 visualize_grid_fovea(num_tiles_fovea, 1, 1);
    const dim3 visualize_grid_blended(num_tiles_blended, 1, 1);
    const dim3 visualize_grid_periphery(num_tiles_periphery, 1, 1);
    if (num_tiles_fovea > 0) {
        kernels::visualize_tile_boundaries_cu<<<visualize_grid_fovea, block, 0, visualize_fovea_stream>>>(
            image,
            make_float3(0.902f, 0.624f, 0.0f),  // orange (Okabe-Ito)
            per_sub_tile_buffers.tile_index_map_partitioned,
            partition_ranges_cpu.fovea_tiles_range.x,
            width,
            height,
            rasterization::config::tile_width_small,
            rasterization::config::tile_height_small,
            grid_large.x,
            to_chw
        );
        CHECK_CUDA(visualization::config::debug_visualization, "visualize_tiles_fovea")
    }
    if (num_tiles_blended > 0) {
        kernels::visualize_tile_boundaries_cu<<<visualize_grid_blended, block, 0, visualize_blended_tiles_stream>>>(
            image,
            make_float3(0.337f, 0.706f, 0.914f),  // light-blue (Okabe-Ito)
            per_sub_tile_buffers.tile_index_map_partitioned,
            partition_ranges_cpu.blended_tiles_range.x,
            width,
            height,
            rasterization::config::tile_width_small,
            rasterization::config::tile_height_small,
            grid_large.x,
            to_chw
        );
        CHECK_CUDA(visualization::config::debug_visualization, "visualize_tiles_blended")
    }
    if (num_tiles_periphery > 0) {
        kernels::visualize_tile_boundaries_cu<<<visualize_grid_periphery, block_large, 0, visualize_periphery_stream>>>(
            image,
            make_float3(0.0f, 0.620f, 0.451f),  // green (Okabe-Ito)
            per_sub_tile_buffers.tile_index_map_partitioned,
            partition_ranges_cpu.periphery_tiles_range.x,
            width,
            height,
            rasterization::config::tile_width_large,
            rasterization::config::tile_height_large,
            grid_large.x,
            to_chw
        );
        CHECK_CUDA(visualization::config::debug_visualization, "visualize_tiles_periphery")
    }

    cudaStreamSynchronize(visualize_fovea_stream);
    cudaStreamSynchronize(visualize_blended_tiles_stream);
    cudaStreamSynchronize(visualize_periphery_stream);
}
