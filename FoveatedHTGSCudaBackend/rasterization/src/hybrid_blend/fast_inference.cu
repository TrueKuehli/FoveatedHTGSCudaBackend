#include "hybrid_blend/fast_inference.h"
#include "hybrid_blend/kernels/fast_inference.cuh"
#include "hybrid_blend/buffer_utils.h"
#include "hybrid_blend/config.h"
#include "shared_kernels.cuh"
#include "rasterization_utils.h"
#include "utils.h"
#include "helper_math.h"
#include <cub/cub.cuh>
#include <functional>
#include <variant>
#include <utility>
#include <type_traits>


template <bool is_lowres_tile, BackgroundModelType background_model, PeripheryInterpolationMode periphery_mode, typename... Args>
void blend_k_templated_periphery_mode(
    const dim3& grid,
    const dim3& block,
    const cudaStream_t stream,
    const int K,
    const int K_blended,
    Args&&... kernel_args)
{
    if (K_blended >= 16) {
        if (K >= 32) htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blend_cu<32, 16, is_lowres_tile, background_model, periphery_mode><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
        else htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blend_cu<16, 16, is_lowres_tile, background_model, periphery_mode><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
    } else if (K_blended >= 8) {
        if (K >= 32) htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blend_cu<32, 8, is_lowres_tile, background_model, periphery_mode><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
        else if (K >= 16) htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blend_cu<16, 8, is_lowres_tile, background_model, periphery_mode><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
        else htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blend_cu<8, 8, is_lowres_tile, background_model, periphery_mode><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
    } else if (K_blended >= 4) {
        if (K >= 32) htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blend_cu<32, 4, is_lowres_tile, background_model, periphery_mode><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
        else if (K >= 16) htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blend_cu<16, 4, is_lowres_tile, background_model, periphery_mode><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
        else if (K >= 8) htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blend_cu<8, 4, is_lowres_tile, background_model, periphery_mode><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
        else htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blend_cu<4, 4, is_lowres_tile, background_model, periphery_mode><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
    } else if (K_blended >= 2) {
        if (K >= 32) htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blend_cu<32, 2, is_lowres_tile, background_model, periphery_mode><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
        else if (K >= 16) htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blend_cu<16, 2, is_lowres_tile, background_model, periphery_mode><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
        else if (K >= 8) htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blend_cu<8, 2, is_lowres_tile, background_model, periphery_mode><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
        else if (K >= 4) htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blend_cu<4, 2, is_lowres_tile, background_model, periphery_mode><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
        else htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blend_cu<2, 2, is_lowres_tile, background_model, periphery_mode><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
    } else {
        if (K >= 32) htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blend_cu<32, 1, is_lowres_tile, background_model, periphery_mode><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
        else if (K >= 16) htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blend_cu<16, 1, is_lowres_tile, background_model, periphery_mode><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
        else if (K >= 8) htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blend_cu<8, 1, is_lowres_tile, background_model, periphery_mode><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
        else if (K >= 4) htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blend_cu<4, 1, is_lowres_tile, background_model, periphery_mode><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
        else if (K >= 2) htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blend_cu<2, 1, is_lowres_tile, background_model, periphery_mode><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
        else htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blend_cu<1, 1, is_lowres_tile, background_model, periphery_mode><<<grid, block, 0, stream>>>(std::forward<Args>(kernel_args)...);
    }
}

template <bool is_lowres_tile, BackgroundModelType background_model, typename... Args>
void blend_k_templated_background_model(
    const dim3& grid,
    const dim3& block,
    const cudaStream_t stream,
    const int K,
    const int K_blended,
    const PeripheryInterpolationMode periphery_mode,
    Args&&... kernel_args)
{
    switch (periphery_mode) {
        case PeripheryInterpolationMode::LINEAR:
            blend_k_templated_periphery_mode<is_lowres_tile, background_model, PeripheryInterpolationMode::LINEAR>(grid, block, stream, K, K_blended, std::forward<Args>(kernel_args)...);
            break;
        case PeripheryInterpolationMode::NEAREST:
        default:
            blend_k_templated_periphery_mode<is_lowres_tile, background_model, PeripheryInterpolationMode::NEAREST>(grid, block, stream, K, K_blended, std::forward<Args>(kernel_args)...);
    }
}

template <bool is_lowres_tile, typename... Args>
void blend_k_templated(
    const dim3& grid,
    const dim3& block,
    const cudaStream_t stream,
    const int K,
    const int K_blended,
    const BackgroundModelType background_model,
    const PeripheryInterpolationMode periphery_mode,
    Args&&... kernel_args)
{
    switch (background_model) {
        case BackgroundModelType::SH:
            blend_k_templated_background_model<is_lowres_tile, BackgroundModelType::SH>(grid, block, stream, K, K_blended, periphery_mode, std::forward<Args>(kernel_args)...);
            break;
        case BackgroundModelType::TEXTURE:
            blend_k_templated_background_model<is_lowres_tile, BackgroundModelType::TEXTURE>(grid, block, stream, K, K_blended, periphery_mode, std::forward<Args>(kernel_args)...);
            break;
        case BackgroundModelType::NONE:
        default:
            blend_k_templated_background_model<is_lowres_tile, BackgroundModelType::NONE>(grid, block, stream, K, K_blended, periphery_mode, std::forward<Args>(kernel_args)...);
    }
}


void htgs_foveated::rasterization::hybrid_blend::fast_inference(
    std::function<char* (size_t)> per_primitive_buffers_func,
    std::function<char* (size_t)> per_tile_buffers_func,
    std::function<char* (size_t)> per_subtile_buffers_func,
    std::function<char* (size_t)> per_instance_buffers_func,
    const float3* positions,
    const float3* scales,
    const float4* rotations,
    const float* opacities,
    const float3* sh_0,
    const float3* sh_rest,
    const float4* M,
    const float4* VPM,
    const float4* VPR_inv,
    const float3* cam_position,
    const float2* gaze_position,
    float* image,
    float* image_final,
    const uint* render_mask,
    const uint* render_mask_area_table,
    const uint* fovea_mask_area_table,
    const float* background_model_data,
    const BackgroundModelType background_model_type,
    const PeripheryInterpolationMode periphery_mode,
    const int K,
    const int n_primitives,
    const int active_sh_bases,
    const int total_sh_bases,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float near_plane,
    const float far_plane,
    const float scale_modifier,
    const bool to_chw,
    const bool blur_periphery,
    const bool anti_aliasing)
{
    cudaMemcpyToSymbol(c_M3, M + 2, sizeof(float4), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_VPM, VPM, 4 * sizeof(float4), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_VPR_inv, VPR_inv, 4 * sizeof(float4), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_cam_position, cam_position, sizeof(float3), 0, cudaMemcpyDeviceToDevice);

    const float2 gaze_position_clamped = make_float2(
        clamp(gaze_position->x, 0.0f, static_cast<float>(width - 1)),
        clamp(gaze_position->y, 0.0f, static_cast<float>(height - 1))
    );
    cudaMemcpyToSymbol(c_gaze_position_cuda, &gaze_position_clamped, sizeof(float2), 0, cudaMemcpyHostToDevice);

    if (background_model_type == BackgroundModelType::SH) {
        cudaMemcpyToSymbol(c_background_sh_coeff, background_model_data, 16 * sizeof(float3), 0, cudaMemcpyDeviceToDevice);
    }

    const dim3 grid_large(div_round_up(width, config::tile_width_large), div_round_up(height, config::tile_height_large), 1);
    const dim3 grid(grid_large.x * config::tile_stride_x, grid_large.y * config::tile_stride_y, 1);
    const dim3 block(config::tile_width_small, config::tile_height_small, 1);
    const int n_tiles_large = grid_large.x * grid_large.y;
    const int n_tiles = grid.x * grid.y;
    const int end_bit = extract_end_bit(n_tiles);

    // TODO: Left and Right eye have different mask, so we need to have separate storage for each eye, or just forgoe constant memory here
    // static uint mask_width = 0;
    // static uint mask_height = 0;
    // if (grid.x != mask_width || grid.y != mask_height) {
    //     mask_width = grid.x;
    //     mask_height = grid.y;
        cudaMemcpyToSymbol(c_render_mask, render_mask, div_round_up(grid_large.x * grid_large.y, 8U), 0, cudaMemcpyDeviceToDevice);
    // }

    // Round gaze to nearest large tile (top left corner of tile)
    const uint2 gaze_position_tiles = make_uint2(
        static_cast<uint>(max(0, min(static_cast<int>(grid.x - 1), (static_cast<int>(gaze_position->x) + config::tile_width_large / 2) / config::tile_width_large))),
        static_cast<uint>(max(0, min(static_cast<int>(grid.y - 1), (static_cast<int>(gaze_position->y) + config::tile_width_large / 2) / config::tile_width_large)))
    );

    constexpr bool store_rgba = true, store_rgb_clamp_info = false;
    char* per_primitive_buffers_blob = per_primitive_buffers_func(required<PerPrimitiveBuffers>(n_primitives, store_rgba, store_rgb_clamp_info));
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives, store_rgba, store_rgb_clamp_info);

    char* per_tile_buffers_blob = per_tile_buffers_func(required<PerTileBuffers>(n_tiles_large));
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles_large);

    // TODO: This is an overallocation; should be optimized to use num_active_tiles
    char* per_sub_tile_buffers_blob = per_subtile_buffers_func(required<PerSubTileBuffers>(n_tiles));
    PerSubTileBuffers per_sub_tile_buffers = PerSubTileBuffers::from_blob(per_sub_tile_buffers_blob, n_tiles);

    static cudaStream_t memset_stream = 0;
    if constexpr (!config::debug_fast_inference) {
        static bool memset_stream_initialized = false;
        if (!memset_stream_initialized) {
            cudaStreamCreate(&memset_stream);
            memset_stream_initialized = true;
        }
        cudaMemsetAsync(per_sub_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles, memset_stream);
    }
    else cudaMemset(per_sub_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles);

    // Build tile index map (so we only need to process [0, num_active_tiles), which we can map back to the "true" tile index)
    shared_kernels::fill_tile_index_num_tiles
            <config::foveation_radius_tiles, config::num_small_tiles_per_large_tile>
            <<<div_round_up(n_tiles_large, config::block_size_create_tile_index_map), config::block_size_create_tile_index_map>>>
    (
        per_tile_buffers.tile_index_map_num_tiles,
        gaze_position_tiles,
        n_tiles_large,
        grid_large.x
    );
    CHECK_CUDA(config::debug_fast_inference, "fill_tile_index_num_tiles")
    cub::DeviceScan::InclusiveSum(
        per_tile_buffers.cub_workspace, per_tile_buffers.cub_workspace_size,
        per_tile_buffers.tile_index_map_num_tiles, per_tile_buffers.tile_index_map_offsets,
        n_tiles_large
    );
    CHECK_CUDA(config::debug_fast_inference, "cub::DeviceScan::InclusiveSum (index_map)")
    shared_kernels::build_tile_index_map
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
    CHECK_CUDA(config::debug_fast_inference, "build_tile_index_map")

    uint num_active_tiles;
    cudaMemcpy(&num_active_tiles, per_tile_buffers.tile_index_map_offsets + n_tiles_large - 1, sizeof(uint), cudaMemcpyDeviceToHost);
    CHECK_CUDA(config::debug_fast_inference, "Fetch num_active_tiles")

    // TODO: Test performance using cub::DevicePartition::If (for three-partition case)
    cub::DeviceRadixSort::SortPairs(
        per_sub_tile_buffers.cub_workspace, per_sub_tile_buffers.cub_workspace_size,
        reinterpret_cast<uint8_t*>(per_sub_tile_buffers.tile_type), reinterpret_cast<uint8_t*>(per_sub_tile_buffers.tile_type_partitioned),
        per_sub_tile_buffers.tile_index_map, per_sub_tile_buffers.tile_index_map_partitioned,
        num_active_tiles, 0, NUM_TILE_TYPE_BITS
    );
    CHECK_CUDA(config::debug_inference, "Sort tiles by type")
    shared_kernels::get_partition_offsets_cu
            <<<div_round_up(static_cast<int>(num_active_tiles), config::block_size_get_partition_offsets), config::block_size_get_partition_offsets>>>
    (
        reinterpret_cast<int*>(per_sub_tile_buffers.partition_offsets),
        per_sub_tile_buffers.tile_type_partitioned,
        num_active_tiles
    );
    CHECK_CUDA(config::debug_inference, "Partition tiles by type")

    PartitionOffsets offsets_cpu;
    cudaMemcpy(&offsets_cpu, per_sub_tile_buffers.partition_offsets, sizeof(PartitionOffsets), cudaMemcpyDeviceToHost);
    const int num_tiles_fovea = offsets_cpu.periphery_tiles_offset;
    const int num_tiles_periphery = offsets_cpu.blended_tiles_offset - offsets_cpu.periphery_tiles_offset;
    const int num_tiles_blended = num_active_tiles - offsets_cpu.blended_tiles_offset;

    const auto preprocess = anti_aliasing ?
        kernels::fast_inference::preprocess_cu<true> :
        kernels::fast_inference::preprocess_cu<false>;
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
        render_mask_area_table,
        fovea_mask_area_table,
        n_primitives,
        grid_large.x,
        grid_large.y,
        active_sh_bases,
        total_sh_bases,
        gaze_position_tiles,
        focal_x,
        focal_y,
        near_plane,
        far_plane,
        scale_modifier
    );
    CHECK_CUDA(config::debug_fast_inference, "preprocess")

    cub::DeviceScan::InclusiveSum(
        per_primitive_buffers.cub_workspace,
        per_primitive_buffers.cub_workspace_size,
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.offset,
        n_primitives
    );
    CHECK_CUDA(config::debug_fast_inference, "cub::DeviceScan::InclusiveSum")

    int n_instances;
    cudaMemcpy(&n_instances, per_primitive_buffers.offset + n_primitives - 1, sizeof(int), cudaMemcpyDeviceToHost);

    std::variant<PerInstanceBuffers<ushort>, PerInstanceBuffers<uint>> buffer_variant;
    if (end_bit <= 16) {
        char* per_instance_buffers_blob = per_instance_buffers_func(required<PerInstanceBuffers<ushort>>(n_instances, end_bit));
        buffer_variant = PerInstanceBuffers<ushort>::from_blob(per_instance_buffers_blob, n_instances, end_bit);
    }
    else {
        char* per_instance_buffers_blob = per_instance_buffers_func(required<PerInstanceBuffers<uint>>(n_instances, end_bit));
        buffer_variant = PerInstanceBuffers<uint>::from_blob(per_instance_buffers_blob, n_instances, end_bit);
    }

    int instance_primitive_indices_selector;
    std::visit([&](auto& per_instance_buffers) {
        using KeyT = std::remove_reference_t<decltype(*per_instance_buffers.keys.Current())>;

        // Ensure random initialized keys cannot overlap with actual valid keys
        cudaMemset(per_instance_buffers.keys.Current(), 255, sizeof(KeyT) * n_instances);
        // compute-sanitizer will complain if the following isn't also executed
        // cudaMemset(per_instance_buffers.primitive_indices.Current(), 255, sizeof(uint) * n_instances);

        shared_kernels::create_instances_cu<KeyT, config::foveation_radius_tiles, config::num_small_tiles_per_large_tile><<<div_round_up(n_primitives, config::block_size_create_instances), config::block_size_create_instances>>>(
            per_primitive_buffers.n_touched_tiles,
            per_primitive_buffers.offset,
            per_primitive_buffers.screen_bounds,
            per_instance_buffers.keys.Current(),
            per_instance_buffers.primitive_indices.Current(),
            make_int2(gaze_position_tiles.x, gaze_position_tiles.y),
            grid_large.x,
            n_primitives
        );
        CHECK_CUDA(config::debug_fast_inference, "create_instances")

        cub::DeviceRadixSort::SortPairs(
            per_instance_buffers.cub_workspace,
            per_instance_buffers.cub_workspace_size,
            per_instance_buffers.keys,
            per_instance_buffers.primitive_indices,
            n_instances,
            0, end_bit
        );
        instance_primitive_indices_selector = per_instance_buffers.primitive_indices.selector;
        CHECK_CUDA(config::debug_fast_inference, "cub::DeviceRadixSort::SortPairs")

        if constexpr (!config::debug_fast_inference) cudaStreamSynchronize(memset_stream);

        if (n_instances > 0) {
            shared_kernels::extract_instance_ranges_cu<KeyT><<<div_round_up(n_instances, config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges>>>(
                per_instance_buffers.keys.Current(),
                per_sub_tile_buffers.instance_ranges,
                n_instances
            );
            CHECK_CUDA(config::debug_fast_inference, "extract_instance_ranges")
        }

        // TODO: Try applying CUDA Graphs to better express dependencies
        static cudaStream_t blend_fovea_stream = 0;
        static cudaStream_t blend_periphery_stream = 0;
        static cudaStream_t blend_blended_tiles_stream = 0;
        static bool blend_streams_initialized = false;
        if (!blend_streams_initialized) {
            cudaStreamCreate(&blend_fovea_stream);
            cudaStreamCreate(&blend_periphery_stream);
            cudaStreamCreate(&blend_blended_tiles_stream);
            blend_streams_initialized = true;
        }

        const dim3 blend_grid_fovea(num_tiles_fovea, 1, 1);
        const dim3 blend_grid_periphery(num_tiles_periphery, 1, 1);
        const dim3 blend_grid_blended(num_tiles_blended, 1, 1);
        // Blended tiles and periphery required to do hole filling, so queue those kernel launches first
        if (num_tiles_blended > 0) {
            blend_k_templated<false>(blend_grid_blended, block, blend_blended_tiles_stream, K, K / 2, background_model_type, periphery_mode,
                per_sub_tile_buffers.tile_index_map_partitioned,
                per_sub_tile_buffers.instance_ranges,
                per_instance_buffers.primitive_indices.Current(),
                per_primitive_buffers.VPMT1,
                per_primitive_buffers.VPMT2,
                per_primitive_buffers.VPMT4,
                per_primitive_buffers.MT3,
                per_primitive_buffers.rgba,
                background_model_data,
                image,
                gaze_position_tiles,
                offsets_cpu.blended_tiles_offset,
                width,
                height,
                grid_large.x,
                to_chw
            );
            CHECK_CUDA(config::debug_fast_inference, "blend_blended_tiles")
        }
        if (num_tiles_periphery > 0) {
            blend_k_templated<true>(blend_grid_periphery, block, blend_periphery_stream, K / 2, K / 2, background_model_type, periphery_mode,
                per_sub_tile_buffers.tile_index_map_partitioned,
                per_sub_tile_buffers.instance_ranges,
                per_instance_buffers.primitive_indices.Current(),
                per_primitive_buffers.VPMT1,
                per_primitive_buffers.VPMT2,
                per_primitive_buffers.VPMT4,
                per_primitive_buffers.MT3,
                per_primitive_buffers.rgba,
                background_model_data,
                image,
                gaze_position_tiles,
                offsets_cpu.periphery_tiles_offset,
                width,
                height,
                grid_large.x,
                to_chw
            );
            CHECK_CUDA(config::debug_inference, "blend_periphery")
        }
        if (num_tiles_fovea > 0) {
            blend_k_templated<false>(blend_grid_fovea, block, blend_fovea_stream, K, K, background_model_type, periphery_mode,
                per_sub_tile_buffers.tile_index_map_partitioned,
                per_sub_tile_buffers.instance_ranges,
                per_instance_buffers.primitive_indices.Current(),
                per_primitive_buffers.VPMT1,
                per_primitive_buffers.VPMT2,
                per_primitive_buffers.VPMT4,
                per_primitive_buffers.MT3,
                per_primitive_buffers.rgba,
                background_model_data,
                blur_periphery ? image_final : image,
                gaze_position_tiles,
                0,  // offset into partitioned tile index map
                width,
                height,
                grid_large.x,
                to_chw
            );
            CHECK_CUDA(config::debug_fast_inference, "blend_fovea")
        }

        cudaStreamSynchronize(blend_blended_tiles_stream);
        if (periphery_mode == PeripheryInterpolationMode::LINEAR) {
            // TODO: Optimization: only launch for tiles that actually have missing pixels (periphery tiles)
            htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::interpolate_missing<<<grid, block, 0, blend_periphery_stream>>>(
                image,
                width,
                height,
                grid_large.x,
                gaze_position_tiles,
                to_chw
            );
        }
        if (blur_periphery) {
            dim3 blend_grid_blur = blend_grid_periphery;
            dim3 blend_grid_copy = blend_grid_blended;
            blend_grid_blur.y = config::num_small_tiles_per_large_tile;
            blend_grid_copy.y = config::num_small_tiles_per_large_tile;
            if (num_tiles_periphery > 0) {
                htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::blur_cu<<<blend_grid_blur, block, 0, blend_periphery_stream>>>(
                    image,
                    image_final,
                    per_sub_tile_buffers.tile_index_map_partitioned,
                    offsets_cpu.periphery_tiles_offset,
                    width,
                    height,
                    grid_large.x,
                    to_chw
                );
                CHECK_CUDA(config::debug_fast_inference, "blur")
            }

            if (num_tiles_blended > 0) {
                htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::copy_pixels<<<blend_grid_copy, block, 0, blend_blended_tiles_stream>>>(
                    image,
                    image_final,
                    per_sub_tile_buffers.tile_index_map_partitioned,
                    offsets_cpu.blended_tiles_offset,
                    width,
                    height,
                    grid_large.x,
                    to_chw
                );
                CHECK_CUDA(config::debug_inference, "copy_pixels")
                cudaStreamSynchronize(blend_blended_tiles_stream);
            }
        }

        cudaStreamSynchronize(blend_fovea_stream);
        cudaStreamSynchronize(blend_periphery_stream);
    }, buffer_variant);

    // // Draw a red dot at the gaze position for visualization
    // const dim3 dot_grid(1, 1, 1);
    // const dim3 dot_block(config::gaze_visualization_width, config::gaze_visualization_width, 1);
    // htgs_foveated::rasterization::hybrid_blend::kernels::fast_inference::visualize_gaze<<<dot_grid, dot_block>>>(
    //     blur_periphery ? image_final : image,
    //     width,
    //     height,
    //     to_chw
    // );
}
