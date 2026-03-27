#include "config.h"
#include "helper_math.h"
#include "inference_stereo.h"
#include "utils.h"
#include "kernels/stereo/inference.cuh"
#include "kernels/stereo/interpolation.cuh"
#include "kernels/stereo/shared_kernels.cuh"
#include "utils/stereo/buffer_utils.h"
#include "utils/rasterization_utils.h"
#include <cub/cub.cuh>
#include <functional>
#include <variant>
#include <utility>
#include <type_traits>


template <bool is_lowres_tile, BackgroundModelType background_model, bool second_camera, typename... Args>
void blend_k_templated_background_model(
    const dim3& grid,
    const dim3& block,
    const cudaStream_t stream,
    const int K,
    Args&&... kernel_args)
{
    if (K >= 32) htgs_foveated::rasterization::kernels::stereo::inference::blend_cu<32, is_lowres_tile, background_model, second_camera><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
    else if (K >= 16) htgs_foveated::rasterization::kernels::stereo::inference::blend_cu<16, is_lowres_tile, background_model, second_camera><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
    else if (K >= 8) htgs_foveated::rasterization::kernels::stereo::inference::blend_cu<8, is_lowres_tile, background_model, second_camera><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
    else if (K >= 4) htgs_foveated::rasterization::kernels::stereo::inference::blend_cu<4, is_lowres_tile, background_model, second_camera><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
    else if (K >= 2) htgs_foveated::rasterization::kernels::stereo::inference::blend_cu<2, is_lowres_tile, background_model, second_camera><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
    else htgs_foveated::rasterization::kernels::stereo::inference::blend_cu<1, is_lowres_tile, background_model, second_camera><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
}

template <bool is_lowres_tile, bool second_camera, typename... Args>
void blend_k_templated(
    const dim3& grid,
    const dim3& block,
    const cudaStream_t stream,
    const int K,
    const BackgroundModelType background_model,
    Args&&... kernel_args)
{
    if (background_model == BackgroundModelType::SH) blend_k_templated_background_model<is_lowres_tile, BackgroundModelType::SH, second_camera>(grid, block, stream, K, std::forward<Args>(kernel_args)...);
    else if (background_model == BackgroundModelType::TEXTURE) blend_k_templated_background_model<is_lowres_tile, BackgroundModelType::TEXTURE, second_camera>(grid, block, stream, K, std::forward<Args>(kernel_args)...);
    else blend_k_templated_background_model<is_lowres_tile, BackgroundModelType::NONE, second_camera>(grid, block, stream, K, std::forward<Args>(kernel_args)...);
}


void htgs_foveated::rasterization::inference_stereo(
        const Buffers& buffers_left,
        const Buffers& buffers_right,
        const float3* positions,
        const float3* scales,
        const float4* rotations,
        const float* opacities,
        const float3* sh_0,
        const float3* sh_rest,
        const BackgroundModel& background_model,
        const Pose& pose_left,
        const Pose& pose_right,
        const Intrinsics& intrinsics_left,
        const Intrinsics& intrinsics_right,
        const Masks& masks_left,
        const Masks& masks_right,
        float* image_left,
        float* image_left_final,
        float* image_right,
        float* image_right_final,
        const int K,
        const int n_primitives,
        const int active_sh_bases,
        const int total_sh_bases,
        const float near_plane,
        const float far_plane,
        const float scale_modifier,
        const bool to_chw,
        const bool blur_periphery,
        const bool anti_aliasing)
{
    cudaMemcpyToSymbol(c_M, pose_left.M, 3 * sizeof(float4), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_M, pose_right.M, 3 * sizeof(float4), sizeof(c_M[0]), cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_VPM, pose_left.VPM, 4 * sizeof(float4), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_VPM, pose_right.VPM, 4 * sizeof(float4), sizeof(c_VPM[0]), cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_VPR_inv, pose_left.VPR_inv, 4 * sizeof(float4), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_VPR_inv, pose_right.VPR_inv, 4 * sizeof(float4), sizeof(c_VPR_inv[0]), cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_cam_position, pose_left.cam_position, sizeof(float3), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_cam_position, pose_right.cam_position, sizeof(float3), sizeof(c_cam_position[0]), cudaMemcpyDeviceToDevice);

    const float2 gaze_position_clamped_left = make_float2(
        clamp(pose_left.gaze_position->x, 0.0f, static_cast<float>(intrinsics_left.width - 1)),
        clamp(pose_left.gaze_position->y, 0.0f, static_cast<float>(intrinsics_left.height - 1))
    );
    const float2 gaze_position_clamped_right = make_float2(
        clamp(pose_right.gaze_position->x, 0.0f, static_cast<float>(intrinsics_right.width - 1)),
        clamp(pose_right.gaze_position->y, 0.0f, static_cast<float>(intrinsics_right.height - 1))
    );
    cudaMemcpyToSymbol(c_gaze_position_cuda, &gaze_position_clamped_left, sizeof(float2), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_gaze_position_cuda, &gaze_position_clamped_right, sizeof(float2), sizeof(c_gaze_position_cuda[0]), cudaMemcpyHostToDevice);

    if (background_model.type == BackgroundModelType::SH) {
        cudaMemcpyToSymbol(c_background_sh_coeff, background_model.data, 16 * sizeof(float3), 0, cudaMemcpyDeviceToDevice);
    }

    const dim3 grid_left_large(div_round_up(intrinsics_left.width, config::tile_width_large), div_round_up(intrinsics_left.height, config::tile_height_large), 1);
    const dim3 grid_right_large(div_round_up(intrinsics_right.width, config::tile_width_large), div_round_up(intrinsics_right.height, config::tile_height_large), 1);
    const dim3 grid_left(grid_left_large.x * config::tile_stride_x, grid_left_large.y * config::tile_stride_y, 1);
    const dim3 grid_right(grid_right_large.x * config::tile_stride_x, grid_right_large.y * config::tile_stride_y, 1);
    const dim3 block(config::tile_width_small, config::tile_height_small, 1);
    const dim3 half_block(config::tile_width_small / 2, config::tile_height_small / 2, 1);
    const int n_tiles_large_left = grid_left_large.x * grid_left_large.y;
    const int n_tiles_left = grid_left.x * grid_left.y;
    const int n_tiles_large_right = grid_right_large.x * grid_right_large.y;
    const int n_tiles_right = grid_right.x * grid_right.y;
    const int end_bit_left = extract_end_bit(n_tiles_left);
    const int end_bit_right = extract_end_bit(n_tiles_right);

    cudaMemcpyToSymbol(c_render_mask, masks_left.render_mask, div_round_up(grid_left_large.x * grid_left_large.y, 8U), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_render_mask, masks_right.render_mask, div_round_up(grid_right_large.x * grid_right_large.y, 8U), sizeof(c_render_mask[0]), cudaMemcpyDeviceToDevice);

    // Round gaze to nearest large tile (top left corner of tile)
    const int2 gaze_position_left_tiles_int = make_int2(
        max(0, min(static_cast<int>(grid_left.x - 1), (static_cast<int>(gaze_position_clamped_left.x) + config::tile_width_large / 2) / config::tile_width_large)),
        max(0, min(static_cast<int>(grid_left.y - 1), (static_cast<int>(gaze_position_clamped_left.y) + config::tile_width_large / 2) / config::tile_width_large))
    );
    const int2 gaze_position_right_tiles_int = make_int2(
        max(0, min(static_cast<int>(grid_right.x - 1), (static_cast<int>(gaze_position_clamped_right.x) + config::tile_width_large / 2) / config::tile_width_large)),
        max(0, min(static_cast<int>(grid_right.y - 1), (static_cast<int>(gaze_position_clamped_right.y) + config::tile_height_large / 2) / config::tile_height_large))
    );
    const float2 gaze_position_left_tiles = make_float2(
        static_cast<float>(gaze_position_left_tiles_int.x),
        static_cast<float>(gaze_position_left_tiles_int.y)
    );
    const float2 gaze_position_right_tiles = make_float2(
        static_cast<float>(gaze_position_right_tiles.x),
        static_cast<float>(gaze_position_right_tiles.y)
    );

    constexpr bool store_rgba = true, store_rgb_clamp_info = false;
    char* per_primitive_buffers_blob_left = buffers_left.per_primitive_buffers_func(required<PerPrimitiveBuffers>(n_primitives, store_rgba, store_rgb_clamp_info));
    char* per_primitive_buffers_blob_right = buffers_right.per_primitive_buffers_func(required<PerPrimitiveBuffers>(n_primitives, store_rgba, store_rgb_clamp_info));
    PerPrimitiveBuffers per_primitive_buffers_left = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob_left, n_primitives, store_rgba, store_rgb_clamp_info);
    PerPrimitiveBuffers per_primitive_buffers_right = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob_right, n_primitives, store_rgba, store_rgb_clamp_info);

    char* per_tile_buffers_blob_left = buffers_left.per_tile_buffers_func(required<PerTileBuffers>(n_tiles_large_left));
    char* per_tile_buffers_blob_right = buffers_right.per_tile_buffers_func(required<PerTileBuffers>(n_tiles_large_right));
    PerTileBuffers per_tile_buffers_left = PerTileBuffers::from_blob(per_tile_buffers_blob_left, n_tiles_large_left);
    PerTileBuffers per_tile_buffers_right = PerTileBuffers::from_blob(per_tile_buffers_blob_right, n_tiles_large_right);

    // TODO: This is an overallocation; should be optimized to use num_active_tiles
    char* per_sub_tile_buffers_blob_left = buffers_left.per_subtile_buffers_func(required<PerSubTileBuffers>(n_tiles_left));
    char* per_sub_tile_buffers_blob_right = buffers_right.per_subtile_buffers_func(required<PerSubTileBuffers>(n_tiles_right));
    PerSubTileBuffers per_sub_tile_buffers_left = PerSubTileBuffers::from_blob(per_sub_tile_buffers_blob_left, n_tiles_left);
    PerSubTileBuffers per_sub_tile_buffers_right = PerSubTileBuffers::from_blob(per_sub_tile_buffers_blob_right, n_tiles_right);

    static cudaStream_t memset_left_stream = 0;
    static cudaStream_t memset_right_stream = 0;
    if constexpr (!config::debug_inference) {
        static bool memset_stream_initialized = false;
        if (!memset_stream_initialized) {
            cudaStreamCreate(&memset_left_stream);
            cudaStreamCreate(&memset_right_stream);
            memset_stream_initialized = true;
        }
        cudaMemsetAsync(per_sub_tile_buffers_left.instance_ranges, 0, sizeof(uint2) * n_tiles_left, memset_left_stream);
        cudaMemsetAsync(per_sub_tile_buffers_left.partition_ranges, 0, sizeof(PartitionRanges), memset_left_stream);
        cudaMemsetAsync(per_sub_tile_buffers_right.instance_ranges, 0, sizeof(uint2) * n_tiles_right, memset_right_stream);
        cudaMemsetAsync(per_sub_tile_buffers_right.partition_ranges, 0, sizeof(PartitionRanges), memset_right_stream);
    } else {
        cudaMemset(per_sub_tile_buffers_left.instance_ranges, 0, sizeof(uint2) * n_tiles_left);
        cudaMemset(per_sub_tile_buffers_left.partition_ranges, 0, sizeof(PartitionRanges));
        cudaMemset(per_sub_tile_buffers_right.instance_ranges, 0, sizeof(uint2) * n_tiles_right);
        cudaMemset(per_sub_tile_buffers_right.partition_ranges, 0, sizeof(PartitionRanges));
    }

    static cudaStream_t preprocess_left_stream = 0;
    static cudaStream_t preprocess_right_stream = 0;
    if constexpr (!config::debug_inference) {
        static bool preprocess_streams_initialized = false;
        if (!preprocess_streams_initialized) {
            cudaStreamCreate(&preprocess_left_stream);
            cudaStreamCreate(&preprocess_right_stream);
            preprocess_streams_initialized = true;
        }
    }

    // Build tile index map (so we only need to process [0, num_active_tiles), which we can map back to the "true" tile index)
    kernels::stereo::shared::fill_tile_index_num_tiles
            <config::foveation_radius_tiles, config::num_small_tiles_per_large_tile, false>
            <<<div_round_up(n_tiles_large_left, config::block_size_create_tile_index_map), config::block_size_create_tile_index_map, 0, preprocess_left_stream>>>
    (
        per_tile_buffers_left.tile_index_map_num_tiles,
        gaze_position_left_tiles,
        n_tiles_large_left,
        grid_left_large.x
    );
    CHECK_CUDA(config::debug_inference, "fill_tile_index_num_tiles left")
    kernels::stereo::shared::fill_tile_index_num_tiles
            <config::foveation_radius_tiles, config::num_small_tiles_per_large_tile, false>
            <<<div_round_up(n_tiles_large_right, config::block_size_create_tile_index_map), config::block_size_create_tile_index_map, 0, preprocess_right_stream>>>
    (
        per_tile_buffers_right.tile_index_map_num_tiles,
        gaze_position_right_tiles,
        n_tiles_large_right,
        grid_right_large.x
    );
    CHECK_CUDA(config::debug_inference, "fill_tile_index_num_tiles right")

    cub::DeviceScan::InclusiveSum(
        per_tile_buffers_left.cub_workspace, per_tile_buffers_left.cub_workspace_size,
        per_tile_buffers_left.tile_index_map_num_tiles, per_tile_buffers_left.tile_index_map_offsets,
        n_tiles_large_left,
        preprocess_left_stream
    );
    CHECK_CUDA(config::debug_inference, "cub::DeviceScan::InclusiveSum (index_map) left")
    cub::DeviceScan::InclusiveSum(
        per_tile_buffers_right.cub_workspace, per_tile_buffers_right.cub_workspace_size,
        per_tile_buffers_right.tile_index_map_num_tiles, per_tile_buffers_right.tile_index_map_offsets,
        n_tiles_large_right,
        preprocess_right_stream
    );
    CHECK_CUDA(config::debug_inference, "cub::DeviceScan::InclusiveSum (index_map) right")

    kernels::stereo::shared::build_tile_index_map
            <config::num_small_tiles_per_large_tile, config::blend_radius_tiles>
            <<<div_round_up(n_tiles_large_left, config::block_size_create_tile_index_map), config::block_size_create_tile_index_map, 0, preprocess_left_stream>>>
    (
        per_sub_tile_buffers_left.tile_index_map,
        per_sub_tile_buffers_left.tile_type,
        per_tile_buffers_left.tile_index_map_num_tiles,
        per_tile_buffers_left.tile_index_map_offsets,
        gaze_position_left_tiles,
        grid_left_large.x,
        n_tiles_large_left
    );
    CHECK_CUDA(config::debug_inference, "build_tile_index_map left")
    kernels::stereo::shared::build_tile_index_map
            <config::num_small_tiles_per_large_tile, config::blend_radius_tiles>
            <<<div_round_up(n_tiles_large_right, config::block_size_create_tile_index_map), config::block_size_create_tile_index_map, 0, preprocess_right_stream>>>
    (
        per_sub_tile_buffers_right.tile_index_map,
        per_sub_tile_buffers_right.tile_type,
        per_tile_buffers_right.tile_index_map_num_tiles,
        per_tile_buffers_right.tile_index_map_offsets,
        gaze_position_right_tiles,
        grid_right_large.x,
        n_tiles_large_right
    );
    CHECK_CUDA(config::debug_inference, "build_tile_index_map right")

    // TODO: Can we get rid of this?
    cudaStreamSynchronize(preprocess_left_stream);
    cudaStreamSynchronize(preprocess_right_stream);

    uint num_active_tiles_left;
    uint num_active_tiles_right;
    cudaMemcpy(&num_active_tiles_left, per_tile_buffers_left.tile_index_map_offsets + n_tiles_large_left - 1, sizeof(uint), cudaMemcpyDeviceToHost);
    CHECK_CUDA(config::debug_inference, "Fetch num_active_tiles left")
    cudaMemcpy(&num_active_tiles_right, per_tile_buffers_right.tile_index_map_offsets + n_tiles_large_right - 1, sizeof(uint), cudaMemcpyDeviceToHost);
    CHECK_CUDA(config::debug_inference, "Fetch num_active_tiles right")

    // TODO: Test performance using cub::DevicePartition::If (for three-partition case)
    cub::DeviceRadixSort::SortPairs(
        per_sub_tile_buffers_left.cub_workspace, per_sub_tile_buffers_left.cub_workspace_size,
        reinterpret_cast<uint8_t*>(per_sub_tile_buffers_left.tile_type), reinterpret_cast<uint8_t*>(per_sub_tile_buffers_left.tile_type_partitioned),
        per_sub_tile_buffers_left.tile_index_map, per_sub_tile_buffers_left.tile_index_map_partitioned,
        num_active_tiles_left, 0, NUM_TILE_TYPE_BITS, preprocess_left_stream
    );
    CHECK_CUDA(config::debug_inference, "Sort tiles by type left")
    cub::DeviceRadixSort::SortPairs(
        per_sub_tile_buffers_right.cub_workspace, per_sub_tile_buffers_right.cub_workspace_size,
        reinterpret_cast<uint8_t*>(per_sub_tile_buffers_right.tile_type), reinterpret_cast<uint8_t*>(per_sub_tile_buffers_right.tile_type_partitioned),
        per_sub_tile_buffers_right.tile_index_map, per_sub_tile_buffers_right.tile_index_map_partitioned,
        num_active_tiles_right, 0, NUM_TILE_TYPE_BITS, preprocess_right_stream
    );
    CHECK_CUDA(config::debug_inference, "Sort tiles by type right")

    if constexpr (!config::debug_inference) {
        cudaStreamSynchronize(memset_left_stream);
        cudaStreamSynchronize(memset_right_stream);
    }
    kernels::stereo::shared::get_partition_ranges_cu
            <<<div_round_up(static_cast<int>(num_active_tiles_left), config::block_size_get_partition_ranges), config::block_size_get_partition_ranges, 0, preprocess_left_stream>>>
    (
        reinterpret_cast<uint2*>(per_sub_tile_buffers_left.partition_ranges),
        per_sub_tile_buffers_left.tile_type_partitioned,
        num_active_tiles_left
    );
    CHECK_CUDA(config::debug_inference, "Partition tiles by type (left)")
    kernels::stereo::shared::get_partition_ranges_cu
            <<<div_round_up(static_cast<int>(num_active_tiles_right), config::block_size_get_partition_ranges), config::block_size_get_partition_ranges, 0, preprocess_right_stream>>>
    (
        reinterpret_cast<uint2*>(per_sub_tile_buffers_right.partition_ranges),
        per_sub_tile_buffers_right.tile_type_partitioned,
        num_active_tiles_right
    );
    CHECK_CUDA(config::debug_inference, "Partition tiles by type (right)")

    // TODO: Can we get rid of this?
    cudaStreamSynchronize(preprocess_left_stream);
    cudaStreamSynchronize(preprocess_right_stream);

    PartitionRanges partition_ranges_cpu_left;
    PartitionRanges partition_ranges_cpu_right;
    cudaMemcpy(&partition_ranges_cpu_left, per_sub_tile_buffers_left.partition_ranges, sizeof(PartitionRanges), cudaMemcpyDeviceToHost);
    CHECK_CUDA(config::debug_inference, "Fetch partition offsets left")
    cudaMemcpy(&partition_ranges_cpu_right, per_sub_tile_buffers_right.partition_ranges, sizeof(PartitionRanges), cudaMemcpyDeviceToHost);
    CHECK_CUDA(config::debug_inference, "Fetch partition offsets right")
    const int num_tiles_fovea_left = partition_ranges_cpu_left.fovea_tiles_range.y - partition_ranges_cpu_left.fovea_tiles_range.x;
    const int num_tiles_periphery_left = partition_ranges_cpu_left.periphery_tiles_range.y - partition_ranges_cpu_left.periphery_tiles_range.x;
    const int num_tiles_blended_left = partition_ranges_cpu_left.blended_tiles_range.y - partition_ranges_cpu_left.blended_tiles_range.x;
    const int num_tiles_fovea_right = partition_ranges_cpu_right.fovea_tiles_range.y - partition_ranges_cpu_right.fovea_tiles_range.x;
    const int num_tiles_periphery_right = partition_ranges_cpu_right.periphery_tiles_range.y - partition_ranges_cpu_right.periphery_tiles_range.x;
    const int num_tiles_blended_right = partition_ranges_cpu_right.blended_tiles_range.y - partition_ranges_cpu_right.blended_tiles_range.x;

    const auto preprocess_left = anti_aliasing ?
        kernels::stereo::inference::preprocess_cu<true, false> :
        kernels::stereo::inference::preprocess_cu<false, false>;
    const auto preprocess_right = anti_aliasing ?
        kernels::stereo::inference::preprocess_cu<true, true> :
        kernels::stereo::inference::preprocess_cu<false, true>;
    preprocess_left<<<div_round_up(n_primitives, config::block_size_preprocess), config::block_size_preprocess, 0, preprocess_left_stream>>>(
        positions,
        scales,
        rotations,
        opacities,
        sh_0,
        sh_rest,
        per_primitive_buffers_left.n_touched_tiles,
        per_primitive_buffers_left.screen_bounds,
        per_primitive_buffers_left.VPMT1,
        per_primitive_buffers_left.VPMT2,
        per_primitive_buffers_left.VPMT4,
        per_primitive_buffers_left.MT3,
        per_primitive_buffers_left.rgba,
        masks_left.render_mask_area_table,
        masks_left.fovea_mask_area_table,
        n_primitives,
        grid_left_large.x,
        grid_left_large.y,
        active_sh_bases,
        total_sh_bases,
        gaze_position_left_tiles_int,
        static_cast<float>(intrinsics_left.width),
        static_cast<float>(intrinsics_left.height),
        intrinsics_left.focal_x,
        intrinsics_left.focal_y,
        intrinsics_left.center_x,
        intrinsics_left.center_y,
        near_plane,
        far_plane,
        scale_modifier
    );
    CHECK_CUDA(config::debug_inference, "preprocess (left)")
    preprocess_right<<<div_round_up(n_primitives, config::block_size_preprocess), config::block_size_preprocess, 0, preprocess_right_stream>>>(
        positions,
        scales,
        rotations,
        opacities,
        sh_0,
        sh_rest,
        per_primitive_buffers_right.n_touched_tiles,
        per_primitive_buffers_right.screen_bounds,
        per_primitive_buffers_right.VPMT1,
        per_primitive_buffers_right.VPMT2,
        per_primitive_buffers_right.VPMT4,
        per_primitive_buffers_right.MT3,
        per_primitive_buffers_right.rgba,
        masks_right.render_mask_area_table,
        masks_right.fovea_mask_area_table,
        n_primitives,
        grid_right_large.x,
        grid_right_large.y,
        active_sh_bases,
        total_sh_bases,
        gaze_position_right_tiles_int,
        static_cast<float>(intrinsics_right.width),
        static_cast<float>(intrinsics_right.height),
        intrinsics_right.focal_x,
        intrinsics_right.focal_y,
        intrinsics_right.center_x,
        intrinsics_right.center_y,
        near_plane,
        far_plane,
        scale_modifier
    );
    CHECK_CUDA(config::debug_inference, "preprocess (right)")

    cub::DeviceScan::InclusiveSum(
        per_primitive_buffers_left.cub_workspace,
        per_primitive_buffers_left.cub_workspace_size,
        per_primitive_buffers_left.n_touched_tiles,
        per_primitive_buffers_left.offset,
        n_primitives,
        preprocess_left_stream
    );
    CHECK_CUDA(config::debug_inference, "cub::DeviceScan::InclusiveSum (left)")
    cub::DeviceScan::InclusiveSum(
        per_primitive_buffers_right.cub_workspace,
        per_primitive_buffers_right.cub_workspace_size,
        per_primitive_buffers_right.n_touched_tiles,
        per_primitive_buffers_right.offset,
        n_primitives,
        preprocess_right_stream
    );
    CHECK_CUDA(config::debug_inference, "cub::DeviceScan::InclusiveSum (right)")

    // TODO: Can we get rid of this?
    cudaStreamSynchronize(preprocess_left_stream);
    cudaStreamSynchronize(preprocess_right_stream);

    int n_instances_left;
    int n_instances_right;
    cudaMemcpy(&n_instances_left, per_primitive_buffers_left.offset + n_primitives - 1, sizeof(int), cudaMemcpyDeviceToHost);
    CHECK_CUDA(config::debug_inference, "Fetch n_instances left")
    cudaMemcpy(&n_instances_right, per_primitive_buffers_right.offset + n_primitives - 1, sizeof(int), cudaMemcpyDeviceToHost);
    CHECK_CUDA(config::debug_inference, "Fetch n_instances right")

    std::variant<PerInstanceBuffers<ushort>, PerInstanceBuffers<uint>> buffer_variant_left;
    std::variant<PerInstanceBuffers<ushort>, PerInstanceBuffers<uint>> buffer_variant_right;
    if (end_bit_left <= 16) {
        char* per_instance_buffers_blob_left = buffers_left.per_instance_buffers_func(required<PerInstanceBuffers<ushort>>(n_instances_left, end_bit_left));
        buffer_variant_left = PerInstanceBuffers<ushort>::from_blob(per_instance_buffers_blob_left, n_instances_left, end_bit_left);
    } else {
        char* per_instance_buffers_blob_left = buffers_left.per_instance_buffers_func(required<PerInstanceBuffers<uint>>(n_instances_left, end_bit_left));
        buffer_variant_left = PerInstanceBuffers<uint>::from_blob(per_instance_buffers_blob_left, n_instances_left, end_bit_left);
    }
    if (end_bit_right <= 16) {
        char* per_instance_buffers_blob_right = buffers_right.per_instance_buffers_func(required<PerInstanceBuffers<ushort>>(n_instances_right, end_bit_right));
        buffer_variant_right = PerInstanceBuffers<ushort>::from_blob(per_instance_buffers_blob_right, n_instances_right, end_bit_right);
    } else {
        char* per_instance_buffers_blob_right = buffers_right.per_instance_buffers_func(required<PerInstanceBuffers<uint>>(n_instances_right, end_bit_right));
        buffer_variant_right = PerInstanceBuffers<uint>::from_blob(per_instance_buffers_blob_right, n_instances_right, end_bit_right);
    }

    std::visit([&](auto& per_instance_buffers_left) {
        using KeyT_left = std::remove_reference_t<decltype(*per_instance_buffers_left.keys.Current())>;
        std::visit([&](auto& per_instance_buffers_right) {
            using KeyT_right = std::remove_reference_t<decltype(*per_instance_buffers_right.keys.Current())>;

            // Ensure random initialized keys cannot overlap with actual valid keys
            if constexpr (!config::debug_inference) {
                cudaMemsetAsync(per_instance_buffers_left.keys.Current(), 255, sizeof(KeyT_left) * n_instances_left, preprocess_left_stream);
                cudaMemsetAsync(per_instance_buffers_right.keys.Current(), 255, sizeof(KeyT_right) * n_instances_right, preprocess_right_stream);
            } else {
                cudaMemset(per_instance_buffers_left.keys.Current(), 255, sizeof(KeyT_left) * n_instances_left);
                cudaMemset(per_instance_buffers_right.keys.Current(), 255, sizeof(KeyT_right) * n_instances_right);
            }

            kernels::stereo::shared::create_instances_cu<KeyT_left, config::foveation_radius_tiles, config::num_small_tiles_per_large_tile, 0><<<div_round_up(n_primitives, config::block_size_create_instances), config::block_size_create_instances, 0, preprocess_left_stream>>>(
                per_primitive_buffers_left.n_touched_tiles,
                per_primitive_buffers_left.offset,
                per_primitive_buffers_left.screen_bounds,
                per_instance_buffers_left.keys.Current(),
                per_instance_buffers_left.primitive_indices.Current(),
                gaze_position_left_tiles,
                grid_left_large.x,
                n_primitives
            );
            CHECK_CUDA(config::debug_inference, "create_instances (left)")
            kernels::stereo::shared::create_instances_cu<KeyT_right, config::foveation_radius_tiles, config::num_small_tiles_per_large_tile, 1><<<div_round_up(n_primitives, config::block_size_create_instances), config::block_size_create_instances, 0, preprocess_right_stream>>>(
                per_primitive_buffers_right.n_touched_tiles,
                per_primitive_buffers_right.offset,
                per_primitive_buffers_right.screen_bounds,
                per_instance_buffers_right.keys.Current(),
                per_instance_buffers_right.primitive_indices.Current(),
                gaze_position_right_tiles,
                grid_right_large.x,
                n_primitives
            );
            CHECK_CUDA(config::debug_inference, "create_instances (right)")

            cub::DeviceRadixSort::SortPairs(
                per_instance_buffers_left.cub_workspace,
                per_instance_buffers_left.cub_workspace_size,
                per_instance_buffers_left.keys,
                per_instance_buffers_left.primitive_indices,
                n_instances_left,
                0, end_bit_left,
                preprocess_left_stream
            );
            CHECK_CUDA(config::debug_inference, "cub::DeviceRadixSort::SortPairs (left)")
            cub::DeviceRadixSort::SortPairs(
                per_instance_buffers_right.cub_workspace,
                per_instance_buffers_right.cub_workspace_size,
                per_instance_buffers_right.keys,
                per_instance_buffers_right.primitive_indices,
                n_instances_right,
                0, end_bit_right,
                preprocess_right_stream
            );
            CHECK_CUDA(config::debug_inference, "cub::DeviceRadixSort::SortPairs (right)")

            if (n_instances_left > 0) {
                kernels::stereo::shared::extract_instance_ranges_cu<KeyT_left><<<div_round_up(n_instances_left, config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges, 0, preprocess_left_stream>>>(
                    per_instance_buffers_left.keys.Current(),
                    per_sub_tile_buffers_left.instance_ranges,
                    n_instances_left
                );
                CHECK_CUDA(config::debug_inference, "extract_instance_ranges (left)")
            }
            if (n_instances_right > 0) {
                kernels::stereo::shared::extract_instance_ranges_cu<KeyT_right><<<div_round_up(n_instances_right, config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges, 0, preprocess_right_stream>>>(
                    per_instance_buffers_right.keys.Current(),
                    per_sub_tile_buffers_right.instance_ranges,
                    n_instances_right
                );
                CHECK_CUDA(config::debug_inference, "extract_instance_ranges (right)")
            }

            static cudaEvent_t preprocess_left_done = 0;
            static cudaEvent_t preprocess_right_done = 0;
            if constexpr (!config::debug_inference) {
                static bool preprocess_events_initialized = false;
                if (!preprocess_events_initialized) {
                    cudaEventCreate(&preprocess_left_done);
                    cudaEventCreate(&preprocess_right_done);
                    preprocess_events_initialized = true;
                }

                cudaEventRecord(preprocess_left_done, preprocess_left_stream);
                cudaEventRecord(preprocess_right_done, preprocess_right_stream);
            }

            static cudaStream_t blend_fovea_stream_left = 0;
            static cudaStream_t blend_periphery_stream_left = 0;
            static cudaStream_t blend_blended_tiles_stream_left = 0;
            static cudaStream_t blend_fovea_stream_right = 0;
            static cudaStream_t blend_periphery_stream_right = 0;
            static cudaStream_t blend_blended_tiles_stream_right = 0;
            if constexpr (!config::debug_inference) {
                static bool blend_streams_initialized = false;
                if (!blend_streams_initialized) {
                    cudaStreamCreate(&blend_fovea_stream_left);
                    cudaStreamCreate(&blend_periphery_stream_left);
                    cudaStreamCreate(&blend_blended_tiles_stream_left);
                    cudaStreamCreate(&blend_fovea_stream_right);
                    cudaStreamCreate(&blend_periphery_stream_right);
                    cudaStreamCreate(&blend_blended_tiles_stream_right);
                    blend_streams_initialized = true;
                }

                cudaStreamWaitEvent(blend_fovea_stream_left, preprocess_left_done, 0);
                cudaStreamWaitEvent(blend_periphery_stream_left, preprocess_left_done, 0);
                cudaStreamWaitEvent(blend_blended_tiles_stream_left, preprocess_left_done, 0);
                cudaStreamWaitEvent(blend_fovea_stream_right, preprocess_right_done, 0);
                cudaStreamWaitEvent(blend_periphery_stream_right, preprocess_right_done, 0);
                cudaStreamWaitEvent(blend_blended_tiles_stream_right, preprocess_right_done, 0);
            }

            const dim3 blend_grid_fovea_left(num_tiles_fovea_left, 1, 1);
            const dim3 blend_grid_periphery_left(num_tiles_periphery_left, 1, 1);
            const dim3 blend_grid_blended_left(num_tiles_blended_left, 1, 1);
            const dim3 blend_grid_fovea_right(num_tiles_fovea_right, 1, 1);
            const dim3 blend_grid_periphery_right(num_tiles_periphery_right, 1, 1);
            const dim3 blend_grid_blended_right(num_tiles_blended_right, 1, 1);

            static cudaEvent_t blend_done[2][3] = {{0,0,0},{0,0,0}};
            if constexpr (!config::debug_inference) {
                static bool blend_events_initialized = false;
                if (!blend_events_initialized) {
                    for (int cam = 0; cam < 2; ++cam) {
                        for (int part = 0; part < 3; ++part) {
                            cudaEventCreate(&blend_done[cam][part]);
                        }
                    }
                    blend_events_initialized = true;
                }
            }
            if (num_tiles_blended_left > 0) {
                blend_k_templated<false, 0>(blend_grid_blended_left, block, blend_blended_tiles_stream_left, K, background_model.type,
                    per_sub_tile_buffers_left.tile_index_map_partitioned,
                    per_sub_tile_buffers_left.instance_ranges,
                    per_instance_buffers_left.primitive_indices.Current(),
                    per_primitive_buffers_left.VPMT1,
                    per_primitive_buffers_left.VPMT2,
                    per_primitive_buffers_left.VPMT4,
                    per_primitive_buffers_left.MT3,
                    per_primitive_buffers_left.rgba,
                    background_model.data,
                    image_left,
                    partition_ranges_cpu_left.blended_tiles_range.x,
                    intrinsics_left.width,
                    intrinsics_left.height,
                    grid_left_large.x,
                    to_chw
                );
                if constexpr (!config::debug_inference) cudaEventRecord(blend_done[0][1], blend_blended_tiles_stream_left);
                CHECK_CUDA(config::debug_inference, "blend_blended_tiles (left)")
            }
            if (num_tiles_blended_right > 0) {
                blend_k_templated<false, 1>(blend_grid_blended_right, block, blend_blended_tiles_stream_right, K, background_model.type,
                    per_sub_tile_buffers_right.tile_index_map_partitioned,
                    per_sub_tile_buffers_right.instance_ranges,
                    per_instance_buffers_right.primitive_indices.Current(),
                    per_primitive_buffers_right.VPMT1,
                    per_primitive_buffers_right.VPMT2,
                    per_primitive_buffers_right.VPMT4,
                    per_primitive_buffers_right.MT3,
                    per_primitive_buffers_right.rgba,
                    background_model.data,
                    image_right,
                    partition_ranges_cpu_right.blended_tiles_range.x,
                    intrinsics_right.width,
                    intrinsics_right.height,
                    grid_right_large.x,
                    to_chw
                );
                if constexpr (!config::debug_inference) cudaEventRecord(blend_done[1][1], blend_blended_tiles_stream_right);
                CHECK_CUDA(config::debug_inference, "blend_blended_tiles (right)")
            }
            if (num_tiles_periphery_left > 0) {
                blend_k_templated<true, 0>(blend_grid_periphery_left, block, blend_periphery_stream_left, K, background_model.type,
                    per_sub_tile_buffers_left.tile_index_map_partitioned,
                    per_sub_tile_buffers_left.instance_ranges,
                    per_instance_buffers_left.primitive_indices.Current(),
                    per_primitive_buffers_left.VPMT1,
                    per_primitive_buffers_left.VPMT2,
                    per_primitive_buffers_left.VPMT4,
                    per_primitive_buffers_left.MT3,
                    per_primitive_buffers_left.rgba,
                    background_model.data,
                    image_left,
                    partition_ranges_cpu_left.periphery_tiles_range.x,
                    intrinsics_left.width,
                    intrinsics_left.height,
                    grid_left_large.x,
                    to_chw
                );
                if constexpr (!config::debug_inference) cudaEventRecord(blend_done[0][2], blend_periphery_stream_left);
                CHECK_CUDA(config::debug_inference, "blend_periphery (left)")
            }
            if (num_tiles_periphery_right > 0) {
                blend_k_templated<true, 1>(blend_grid_periphery_right, block, blend_periphery_stream_right, K, background_model.type,
                    per_sub_tile_buffers_right.tile_index_map_partitioned,
                    per_sub_tile_buffers_right.instance_ranges,
                    per_instance_buffers_right.primitive_indices.Current(),
                    per_primitive_buffers_right.VPMT1,
                    per_primitive_buffers_right.VPMT2,
                    per_primitive_buffers_right.VPMT4,
                    per_primitive_buffers_right.MT3,
                    per_primitive_buffers_right.rgba,
                    background_model.data,
                    image_right,
                    partition_ranges_cpu_right.periphery_tiles_range.x,
                    intrinsics_right.width,
                    intrinsics_right.height,
                    grid_right_large.x,
                    to_chw
                );
                if constexpr (!config::debug_inference) cudaEventRecord(blend_done[1][2], blend_periphery_stream_right);
                CHECK_CUDA(config::debug_inference, "blend_periphery (right)")
            }
            if (num_tiles_fovea_left > 0) {
                blend_k_templated<false, 0>(blend_grid_fovea_left, block, blend_fovea_stream_left, K, background_model.type,
                    per_sub_tile_buffers_left.tile_index_map_partitioned,
                    per_sub_tile_buffers_left.instance_ranges,
                    per_instance_buffers_left.primitive_indices.Current(),
                    per_primitive_buffers_left.VPMT1,
                    per_primitive_buffers_left.VPMT2,
                    per_primitive_buffers_left.VPMT4,
                    per_primitive_buffers_left.MT3,
                    per_primitive_buffers_left.rgba,
                    background_model.data,
                    image_left_final,
                    partition_ranges_cpu_left.fovea_tiles_range.x,
                    intrinsics_left.width,
                    intrinsics_left.height,
                    grid_left_large.x,
                    to_chw
                );
                if constexpr (!config::debug_inference) cudaEventRecord(blend_done[0][0], blend_fovea_stream_left);
                CHECK_CUDA(config::debug_inference, "blend_fovea (left)")
            }
            if (num_tiles_fovea_right > 0) {
                blend_k_templated<false, 1>(blend_grid_fovea_right, block, blend_fovea_stream_right, K, background_model.type,
                    per_sub_tile_buffers_right.tile_index_map_partitioned,
                    per_sub_tile_buffers_right.instance_ranges,
                    per_instance_buffers_right.primitive_indices.Current(),
                    per_primitive_buffers_right.VPMT1,
                    per_primitive_buffers_right.VPMT2,
                    per_primitive_buffers_right.VPMT4,
                    per_primitive_buffers_right.MT3,
                    per_primitive_buffers_right.rgba,
                    background_model.data,
                    image_right_final,
                    partition_ranges_cpu_right.fovea_tiles_range.x,
                    intrinsics_right.width,
                    intrinsics_right.height,
                    grid_right_large.x,
                    to_chw
                );
                if constexpr (!config::debug_inference) cudaEventRecord(blend_done[1][0], blend_fovea_stream_right);
                CHECK_CUDA(config::debug_inference, "blend_fovea (right)")
            }

            if (blur_periphery) {
                dim3 blend_grid_blur_left = blend_grid_periphery_left;
                dim3 blend_grid_blur_right = blend_grid_periphery_right;
                dim3 blend_grid_blur_blended_left = blend_grid_blended_left;
                dim3 blend_grid_blur_blended_right = blend_grid_blended_right;
                blend_grid_blur_left.y = config::tile_stride_x;
                blend_grid_blur_left.z = config::tile_stride_y;
                blend_grid_blur_right.y = config::tile_stride_x;
                blend_grid_blur_right.z = config::tile_stride_y;
                blend_grid_blur_blended_left.y = config::tile_stride_x;
                blend_grid_blur_blended_left.z = config::tile_stride_y;
                blend_grid_blur_blended_right.y = config::tile_stride_x;
                blend_grid_blur_blended_right.z = config::tile_stride_y;

                                if (num_tiles_periphery_left > 0) {
                    // Wait for blended tiles blending to be done
                    if constexpr (!config::debug_inference) cudaStreamWaitEvent(blend_periphery_stream_left, blend_done[0][1], 0);
                    kernels::stereo::interpolation::interpolate_and_blur<<<blend_grid_blur_left, block, 0, blend_periphery_stream_left>>>(
                        image_left_final,
                        image_left,
                        per_sub_tile_buffers_left.tile_index_map_partitioned,
                        partition_ranges_cpu_left.periphery_tiles_range.x,
                        intrinsics_left.width,
                        intrinsics_left.height,
                        grid_left_large.x,
                        to_chw
                    );
                    CHECK_CUDA(config::debug_inference, "blur (left)")
                }
                if (num_tiles_periphery_right > 0) {
                    // Wait for blended tiles blending to be done
                    if constexpr (!config::debug_inference) cudaStreamWaitEvent(blend_periphery_stream_right, blend_done[1][1], 0);
                    kernels::stereo::interpolation::interpolate_and_blur<<<blend_grid_blur_right, block, 0, blend_periphery_stream_right>>>(
                        image_right_final,
                        image_right,
                        per_sub_tile_buffers_right.tile_index_map_partitioned,
                        partition_ranges_cpu_right.periphery_tiles_range.x,
                        intrinsics_right.width,
                        intrinsics_right.height,
                        grid_right_large.x,
                        to_chw
                    );
                    CHECK_CUDA(config::debug_inference, "blur (right)")
                }
                if (num_tiles_blended_left > 0) {
                    // Wait for fovea + periphery blending to be done
                    if constexpr (!config::debug_inference) cudaStreamWaitEvent(blend_blended_tiles_stream_left, blend_done[0][0], 0);
                    if constexpr (!config::debug_inference) cudaStreamWaitEvent(blend_blended_tiles_stream_left, blend_done[0][2], 0);
                    kernels::stereo::interpolation::interpolate_and_blur_blended<0><<<blend_grid_blur_blended_left, half_block, 0, blend_blended_tiles_stream_left>>>(
                        image_left_final,
                        image_left,
                        per_sub_tile_buffers_left.tile_index_map_partitioned,
                        partition_ranges_cpu_left.blended_tiles_range.x,
                        intrinsics_left.width,
                        intrinsics_left.height,
                        grid_left_large.x,
                        to_chw
                    );
                    CHECK_CUDA(config::debug_inference, "blur_blended (left)")
                }
                if (num_tiles_blended_right > 0) {
                    // Wait for fovea + periphery blending to be done
                    if constexpr (!config::debug_inference) cudaStreamWaitEvent(blend_blended_tiles_stream_right, blend_done[1][0], 0);
                    if constexpr (!config::debug_inference) cudaStreamWaitEvent(blend_blended_tiles_stream_right, blend_done[1][2], 0);
                    kernels::stereo::interpolation::interpolate_and_blur_blended<1><<<blend_grid_blur_blended_right, half_block, 0, blend_blended_tiles_stream_right>>>(
                        image_right_final,
                        image_right,
                        per_sub_tile_buffers_right.tile_index_map_partitioned,
                        partition_ranges_cpu_right.blended_tiles_range.x,
                        intrinsics_right.width,
                        intrinsics_right.height,
                        grid_right_large.x,
                        to_chw
                    );
                    CHECK_CUDA(config::debug_inference, "blur_blended (right)")
                }
            } else {
                // TODO: Implement non-blur path
            }

            cudaStreamSynchronize(blend_fovea_stream_left);
            cudaStreamSynchronize(blend_fovea_stream_right);
            cudaStreamSynchronize(blend_blended_tiles_stream_left);
            cudaStreamSynchronize(blend_blended_tiles_stream_right);
            cudaStreamSynchronize(blend_periphery_stream_left);
            cudaStreamSynchronize(blend_periphery_stream_right);
        }, buffer_variant_right);
    }, buffer_variant_left);
}
