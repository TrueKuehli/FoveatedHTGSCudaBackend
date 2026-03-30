#include "config.h"
#include "helper_math.h"
#include "inference.h"
#include "utils.h"
#include "kernels/inference.cuh"
#include "kernels/interpolation.cuh"
#include "kernels/shared_kernels.cuh"
#include "utils/buffer_utils.h"
#include "utils/rasterization_utils.h"
#include <cub/cub.cuh>
#include <functional>
#include <variant>
#include <utility>
#include <type_traits>


template <bool is_lowres_tile, BackgroundModelType background_model, typename... Args>
static void blend_k_templated_background_model(
    const dim3& grid,
    const dim3& block,
    const cudaStream_t stream,
    const int K,
    Args&&... kernel_args)
{
    if (K >= 32) htgs_foveated::rasterization::kernels::inference::blend_cu<32, is_lowres_tile, background_model, 0><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
    else if (K >= 16) htgs_foveated::rasterization::kernels::inference::blend_cu<16, is_lowres_tile, background_model, 0><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
    else if (K >= 8) htgs_foveated::rasterization::kernels::inference::blend_cu<8, is_lowres_tile, background_model, 0><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
    else if (K >= 4) htgs_foveated::rasterization::kernels::inference::blend_cu<4, is_lowres_tile, background_model, 0><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
    else if (K >= 2) htgs_foveated::rasterization::kernels::inference::blend_cu<2, is_lowres_tile, background_model, 0><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
    else htgs_foveated::rasterization::kernels::inference::blend_cu<1, is_lowres_tile, background_model, 0><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
}

template <bool is_lowres_tile, typename... Args>
static void blend_k_templated(
    const dim3& grid,
    const dim3& block,
    const cudaStream_t stream,
    const int K,
    const BackgroundModelType background_model,
    Args&&... kernel_args)
{
    if (background_model == BackgroundModelType::SH) blend_k_templated_background_model<is_lowres_tile, BackgroundModelType::SH>(grid, block, stream, K, std::forward<Args>(kernel_args)...);
    else if (background_model == BackgroundModelType::TEXTURE) blend_k_templated_background_model<is_lowres_tile, BackgroundModelType::TEXTURE>(grid, block, stream, K, std::forward<Args>(kernel_args)...);
    else blend_k_templated_background_model<is_lowres_tile, BackgroundModelType::NONE>(grid, block, stream, K, std::forward<Args>(kernel_args)...);
}


void htgs_foveated::rasterization::inference(
        const Buffers& buffers,
        const float3* positions,
        const float3* scales,
        const float4* rotations,
        const float* opacities,
        const float3* sh_0,
        const float3* sh_rest,
        const BackgroundModel& background_model,
        const Pose& pose,
        const Intrinsics& intrinsics,
        const Masks& masks,
        float* image,
        float* image_final,
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
    cudaMemcpyToSymbol(c_M, pose.M, 3 * sizeof(float4), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_VPM, pose.VPM, 4 * sizeof(float4), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_VPR_inv, pose.VPR_inv, 4 * sizeof(float4), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_cam_position, pose.cam_position, sizeof(float3), 0, cudaMemcpyDeviceToDevice);

    const float2 gaze_position_clamped = make_float2(
        clamp(pose.gaze_position->x, 0.0f, static_cast<float>(intrinsics.width - 1)),
        clamp(pose.gaze_position->y, 0.0f, static_cast<float>(intrinsics.height - 1))
    );
    cudaMemcpyToSymbol(c_gaze_position_cuda, &gaze_position_clamped, sizeof(float2), 0, cudaMemcpyHostToDevice);

    if (background_model.type == BackgroundModelType::SH) {
        cudaMemcpyToSymbol(c_background_sh_coeff, background_model.data, 16 * sizeof(float3), 0, cudaMemcpyDeviceToDevice);
    }

    const dim3 grid_large(div_round_up(intrinsics.width, config::tile_width_large), div_round_up(intrinsics.height, config::tile_height_large), 1);
    const dim3 grid(grid_large.x * config::tile_stride_x, grid_large.y * config::tile_stride_y, 1);
    const dim3 block(config::tile_width_small, config::tile_height_small, 1);
    const dim3 half_block(config::tile_width_small / 2, config::tile_height_small / 2, 1);
    const int n_tiles_large = grid_large.x * grid_large.y;
    const int n_tiles = grid.x * grid.y;

    cudaMemcpyToSymbol(c_render_mask, masks.render_mask, div_round_up(grid_large.x * grid_large.y, 8U), 0, cudaMemcpyDeviceToDevice);
    const int end_bit = extract_end_bit(n_tiles + 1);

    // Round gaze to nearest large tile (top left corner of tile)
    const int2 gaze_position_tiles_int = make_int2(
        max(0, min(static_cast<int>(grid.x - 1), (static_cast<int>(gaze_position_clamped.x) + config::tile_width_large / 2) / config::tile_width_large)),
        max(0, min(static_cast<int>(grid.y - 1), (static_cast<int>(gaze_position_clamped.y) + config::tile_width_large / 2) / config::tile_width_large))
    );
    const float2 gaze_position_tiles = make_float2(
        static_cast<float>(gaze_position_tiles_int.x),
        static_cast<float>(gaze_position_tiles_int.y)
    );

    constexpr bool store_rgba = true, store_rgb_clamp_info = false;
    char* per_primitive_buffers_blob = buffers.per_primitive_buffers_func(required<PerPrimitiveBuffers>(n_primitives, store_rgba, store_rgb_clamp_info));
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives, store_rgba, store_rgb_clamp_info);

    char* per_tile_buffers_blob = buffers.per_tile_buffers_func(required<PerTileBuffers>(n_tiles_large));
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles_large);

    // TODO: This is an overallocation; should be optimized to use num_active_tiles
    char* per_sub_tile_buffers_blob = buffers.per_subtile_buffers_func(required<PerSubTileBuffers>(n_tiles));
    PerSubTileBuffers per_sub_tile_buffers = PerSubTileBuffers::from_blob(per_sub_tile_buffers_blob, n_tiles);

    static cudaStream_t memset_stream = 0;
    if constexpr (!config::debug_inference) {
        static bool memset_stream_initialized = false;
        if (!memset_stream_initialized) {
            cudaStreamCreate(&memset_stream);
            memset_stream_initialized = true;
        }
        cudaMemsetAsync(per_sub_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles, memset_stream);
        cudaMemsetAsync(per_sub_tile_buffers.partition_ranges, 0, sizeof(PartitionRanges), memset_stream);
    } else {
        cudaMemset(per_sub_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles);
        cudaMemset(per_sub_tile_buffers.partition_ranges, 0, sizeof(PartitionRanges));
    }

    // Build tile index map (so we only need to process [0, num_active_tiles), which we can map back to the "true" tile index)
    kernels::shared::fill_tile_index_num_tiles
            <config::foveation_radius_tiles, config::num_small_tiles_per_large_tile, 0>
            <<<div_round_up(n_tiles_large, config::block_size_create_tile_index_map), config::block_size_create_tile_index_map>>>
    (
        per_tile_buffers.tile_index_map_num_tiles,
        gaze_position_tiles,
        n_tiles_large,
        grid_large.x
    );
    CHECK_CUDA(config::debug_inference, "fill_tile_index_num_tiles")
    cub::DeviceScan::InclusiveSum(
        per_tile_buffers.cub_workspace, per_tile_buffers.cub_workspace_size,
        per_tile_buffers.tile_index_map_num_tiles, per_tile_buffers.tile_index_map_offsets,
        n_tiles_large
    );
    CHECK_CUDA(config::debug_inference, "cub::DeviceScan::InclusiveSum (index_map)")
    kernels::shared::build_tile_index_map
            <config::num_small_tiles_per_large_tile, config::blend_radius_tiles>
            <<<div_round_up(n_tiles_large, config::block_size_create_tile_index_map), config::block_size_create_tile_index_map>>>
    (
        per_sub_tile_buffers.tile_index_map,
        per_sub_tile_buffers.tile_type,
        per_tile_buffers.tile_index_map_num_tiles,
        per_tile_buffers.tile_index_map_offsets,
        gaze_position_tiles,
        grid_large.x,
        n_tiles_large
    );
    CHECK_CUDA(config::debug_inference, "build_tile_index_map")

    uint num_active_tiles;
    cudaMemcpy(&num_active_tiles, per_tile_buffers.tile_index_map_offsets + n_tiles_large - 1, sizeof(uint), cudaMemcpyDeviceToHost);
    CHECK_CUDA(config::debug_inference, "Fetch num_active_tiles")

    // TODO: Test performance using cub::DevicePartition::If (for three-partition case)
    cub::DeviceRadixSort::SortPairs(
        per_sub_tile_buffers.cub_workspace, per_sub_tile_buffers.cub_workspace_size,
        reinterpret_cast<uint8_t*>(per_sub_tile_buffers.tile_type), reinterpret_cast<uint8_t*>(per_sub_tile_buffers.tile_type_partitioned),
        per_sub_tile_buffers.tile_index_map, per_sub_tile_buffers.tile_index_map_partitioned,
        num_active_tiles, 0, NUM_TILE_TYPE_BITS
    );
    CHECK_CUDA(config::debug_inference, "Sort tiles by type")
    if constexpr (!config::debug_inference) cudaStreamSynchronize(memset_stream);
    kernels::shared::get_partition_ranges_cu
            <<<div_round_up(static_cast<int>(num_active_tiles), config::block_size_get_partition_ranges), config::block_size_get_partition_ranges>>>
    (
        reinterpret_cast<uint2*>(per_sub_tile_buffers.partition_ranges),
        per_sub_tile_buffers.tile_type_partitioned,
        num_active_tiles
    );
    CHECK_CUDA(config::debug_inference, "Partition tiles by type")

    PartitionRanges partition_ranges_cpu;
    cudaMemcpy(&partition_ranges_cpu, per_sub_tile_buffers.partition_ranges, sizeof(PartitionRanges), cudaMemcpyDeviceToHost);
    const int num_tiles_fovea = partition_ranges_cpu.fovea_tiles_range.y - partition_ranges_cpu.fovea_tiles_range.x;
    const int num_tiles_periphery = partition_ranges_cpu.periphery_tiles_range.y - partition_ranges_cpu.periphery_tiles_range.x;
    const int num_tiles_blended = partition_ranges_cpu.blended_tiles_range.y - partition_ranges_cpu.blended_tiles_range.x;

    const auto preprocess = anti_aliasing ?
        kernels::inference::preprocess_cu<true, 0> :
        kernels::inference::preprocess_cu<false, 0>;
    preprocess<<<div_round_up(n_primitives, config::block_size_preprocess), config::block_size_preprocess>>>(
        positions,
        scales,
        rotations,
        opacities,
        sh_0,
        sh_rest,
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.screen_bounds,
        per_primitive_buffers.VPMT1,
        per_primitive_buffers.VPMT2,
        per_primitive_buffers.VPMT4,
        per_primitive_buffers.MT3,
        per_primitive_buffers.rgba,
        masks.render_mask_area_table,
        masks.fovea_mask_area_table,
        n_primitives,
        grid_large.x,
        grid_large.y,
        active_sh_bases,
        total_sh_bases,
        gaze_position_tiles_int,
        static_cast<float>(intrinsics.width),
        static_cast<float>(intrinsics.height),
        intrinsics.focal_x,
        intrinsics.focal_y,
        intrinsics.center_x,
        intrinsics.center_y,
        near_plane,
        far_plane,
        scale_modifier
    );
    CHECK_CUDA(config::debug_inference, "preprocess")

    cub::DeviceScan::InclusiveSum(
        per_primitive_buffers.cub_workspace,
        per_primitive_buffers.cub_workspace_size,
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.offset,
        n_primitives
    );
    CHECK_CUDA(config::debug_inference, "cub::DeviceScan::InclusiveSum")

    int n_instances;
    cudaMemcpy(&n_instances, per_primitive_buffers.offset + n_primitives - 1, sizeof(int), cudaMemcpyDeviceToHost);

    std::variant<PerInstanceBuffers<ushort>, PerInstanceBuffers<uint>> buffer_variant;
    if (end_bit <= 16) {
        char* per_instance_buffers_blob = buffers.per_instance_buffers_func(required<PerInstanceBuffers<ushort>>(n_instances, end_bit));
        buffer_variant = PerInstanceBuffers<ushort>::from_blob(per_instance_buffers_blob, n_instances, end_bit);
    }
    else {
        char* per_instance_buffers_blob = buffers.per_instance_buffers_func(required<PerInstanceBuffers<uint>>(n_instances, end_bit));
        buffer_variant = PerInstanceBuffers<uint>::from_blob(per_instance_buffers_blob, n_instances, end_bit);
    }

    std::visit([&](auto& per_instance_buffers) {
        using KeyT = std::remove_reference_t<decltype(*per_instance_buffers.keys.Current())>;

        // Ensure random initialized keys cannot overlap with actual valid keys
        cudaMemset(per_instance_buffers.keys.Current(), 255, sizeof(KeyT) * n_instances);

        kernels::shared::create_instances_cu<KeyT, config::foveation_radius_tiles, config::num_small_tiles_per_large_tile, 0><<<div_round_up(n_primitives, config::block_size_create_instances), config::block_size_create_instances>>>(
            per_primitive_buffers.n_touched_tiles,
            per_primitive_buffers.offset,
            per_primitive_buffers.screen_bounds,
            per_instance_buffers.keys.Current(),
            per_instance_buffers.primitive_indices.Current(),
            gaze_position_tiles,
            grid_large.x,
            n_primitives
        );
        CHECK_CUDA(config::debug_inference, "create_instances")

        cub::DeviceRadixSort::SortPairs(
            per_instance_buffers.cub_workspace,
            per_instance_buffers.cub_workspace_size,
            per_instance_buffers.keys,
            per_instance_buffers.primitive_indices,
            n_instances,
            0, end_bit
        );
        CHECK_CUDA(config::debug_inference, "cub::DeviceRadixSort::SortPairs")

        if (n_instances > 0) {
            kernels::shared::extract_instance_ranges_cu<KeyT><<<div_round_up(n_instances, config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges>>>(
                per_instance_buffers.keys.Current(),
                per_sub_tile_buffers.instance_ranges,
                n_instances
            );
            CHECK_CUDA(config::debug_inference, "extract_instance_ranges")
        }

        // Ensure pre-processing is fully done before blending
        static cudaEvent_t preprocess_done = 0;
        if constexpr (!config::debug_inference) {
            static bool preprocess_events_initialized = false;
            if (!preprocess_events_initialized) {
                cudaEventCreate(&preprocess_done);
                preprocess_events_initialized = true;
            }

            cudaEventRecord(preprocess_done, 0);
        }

        static cudaStream_t blend_fovea_stream = 0;
        static cudaStream_t blend_periphery_stream = 0;
        static cudaStream_t blend_blended_tiles_stream = 0;
        static bool blend_streams_initialized = false;
        if constexpr (!config::debug_inference) {
            if (!blend_streams_initialized) {
                cudaStreamCreate(&blend_fovea_stream);
                cudaStreamCreate(&blend_periphery_stream);
                cudaStreamCreate(&blend_blended_tiles_stream);
                blend_streams_initialized = true;
            }

            cudaStreamWaitEvent(blend_fovea_stream, preprocess_done, 0);
            cudaStreamWaitEvent(blend_periphery_stream, preprocess_done, 0);
            cudaStreamWaitEvent(blend_blended_tiles_stream, preprocess_done, 0);
        }

        const dim3 blend_grid_fovea(num_tiles_fovea, 1, 1);
        const dim3 blend_grid_periphery(num_tiles_periphery, 1, 1);
        const dim3 blend_grid_blended(num_tiles_blended, 1, 1);
        static struct {
            cudaEvent_t fovea;
            cudaEvent_t blended;
            cudaEvent_t periphery;
        } blend_done = {0,0,0};
        if constexpr (!config::debug_inference) {
            static bool blend_events_initialized = false;
            if (!blend_events_initialized) {
                cudaEventCreate(&blend_done.fovea);
                cudaEventCreate(&blend_done.blended);
                cudaEventCreate(&blend_done.periphery);
                blend_events_initialized = true;
            }
        }

        // Blended tiles and periphery required to do hole filling, so queue those kernel launches first
        if (num_tiles_blended > 0) {
            blend_k_templated<false>(blend_grid_blended, block, blend_blended_tiles_stream, K, background_model.type,
                per_sub_tile_buffers.tile_index_map_partitioned,
                per_sub_tile_buffers.instance_ranges,
                per_instance_buffers.primitive_indices.Current(),
                per_primitive_buffers.VPMT1,
                per_primitive_buffers.VPMT2,
                per_primitive_buffers.VPMT4,
                per_primitive_buffers.MT3,
                per_primitive_buffers.rgba,
                background_model.data,
                image,
                partition_ranges_cpu.blended_tiles_range.x,
                intrinsics.width,
                intrinsics.height,
                grid_large.x,
                to_chw
            );
            if constexpr (!config::debug_inference) cudaEventRecord(blend_done.blended, blend_blended_tiles_stream);
            CHECK_CUDA(config::debug_inference, "blend_blended_tiles")
        }
        if (num_tiles_periphery > 0) {
            blend_k_templated<true>(blend_grid_periphery, block, blend_periphery_stream, K, background_model.type,
                per_sub_tile_buffers.tile_index_map_partitioned,
                per_sub_tile_buffers.instance_ranges,
                per_instance_buffers.primitive_indices.Current(),
                per_primitive_buffers.VPMT1,
                per_primitive_buffers.VPMT2,
                per_primitive_buffers.VPMT4,
                per_primitive_buffers.MT3,
                per_primitive_buffers.rgba,
                background_model.data,
                image,
                partition_ranges_cpu.periphery_tiles_range.x,
                intrinsics.width,
                intrinsics.height,
                grid_large.x,
                to_chw
            );
            if constexpr (!config::debug_inference) cudaEventRecord(blend_done.periphery, blend_periphery_stream);
            CHECK_CUDA(config::debug_inference, "blend_periphery")
        }
        if (num_tiles_fovea > 0) {
            blend_k_templated<false>(blend_grid_fovea, block, blend_fovea_stream, K, background_model.type,
                per_sub_tile_buffers.tile_index_map_partitioned,
                per_sub_tile_buffers.instance_ranges,
                per_instance_buffers.primitive_indices.Current(),
                per_primitive_buffers.VPMT1,
                per_primitive_buffers.VPMT2,
                per_primitive_buffers.VPMT4,
                per_primitive_buffers.MT3,
                per_primitive_buffers.rgba,
                background_model.data,
                image_final,
                partition_ranges_cpu.fovea_tiles_range.x,
                intrinsics.width,
                intrinsics.height,
                grid_large.x,
                to_chw
            );
            if constexpr (!config::debug_inference) cudaEventRecord(blend_done.fovea, blend_fovea_stream);
            CHECK_CUDA(config::debug_inference, "blend_fovea")
        }

        if (blur_periphery) {
            dim3 blend_grid_blur = blend_grid_periphery;
            dim3 blend_grid_blur_blended = blend_grid_blended;
            blend_grid_blur.y = config::tile_stride_x;
            blend_grid_blur.z = config::tile_stride_y;
            blend_grid_blur_blended.y = config::tile_stride_x;
            blend_grid_blur_blended.z = config::tile_stride_y;

            if (num_tiles_periphery > 0) {
                // Wait for blended tiles blending to be done
                if constexpr (!config::debug_inference) cudaStreamWaitEvent(blend_periphery_stream, blend_done.blended, 0);
                kernels::interpolation::interpolate_and_blur<<<blend_grid_blur, block, 0, blend_periphery_stream>>>(
                    image_final,
                    image,
                    per_sub_tile_buffers.tile_index_map_partitioned,
                    partition_ranges_cpu.periphery_tiles_range.x,
                    intrinsics.width,
                    intrinsics.height,
                    grid_large.x,
                    to_chw
                );
                CHECK_CUDA(config::debug_inference, "blur")
            }

            if (num_tiles_blended > 0) {
                // Wait for fovea + periphery blending to be done
                if constexpr (!config::debug_inference) cudaStreamWaitEvent(blend_blended_tiles_stream, blend_done.fovea, 0);
                if constexpr (!config::debug_inference) cudaStreamWaitEvent(blend_blended_tiles_stream, blend_done.periphery, 0);
                kernels::interpolation::interpolate_and_blur_blended<0><<<blend_grid_blur_blended, half_block, 0, blend_blended_tiles_stream>>>(
                    image_final,
                    image,
                    per_sub_tile_buffers.tile_index_map_partitioned,
                    partition_ranges_cpu.blended_tiles_range.x,
                    intrinsics.width,
                    intrinsics.height,
                    grid_large.x,
                    to_chw
                );
                CHECK_CUDA(config::debug_inference, "blur_blended")
            }
        } else {
            // TODO: Implement non-blur path
        }

        cudaStreamSynchronize(blend_fovea_stream);
        cudaStreamSynchronize(blend_periphery_stream);
        cudaStreamSynchronize(blend_blended_tiles_stream);
    }, buffer_variant);
}
