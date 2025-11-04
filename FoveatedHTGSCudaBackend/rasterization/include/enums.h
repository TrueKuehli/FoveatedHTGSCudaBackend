#pragma once

#include <bit>
#include <cstdint>


enum class TileType : uint8_t {
    FOVEA = 0,
    PERIPHERY = 1,
    BLENDED = 2,
    _NUM_TYPES = 3
};

constexpr int NUM_TILE_TYPE_BITS = std::bit_width(static_cast<uint8_t>(static_cast<uint8_t>(TileType::_NUM_TYPES) - 1));