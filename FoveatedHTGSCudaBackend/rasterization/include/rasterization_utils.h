#pragma once

enum class PeripheryInterpolationMode {
    NEAREST = 0,
    LINEAR = 1,
};


enum class BackgroundModelType {
    NONE = 0,
    SH = 1,
    TEXTURE = 2,
};


inline __host__ int extract_end_bit(
    uint n)
{
    int leading_zeros = 0;
    if ((n & 0xffff0000) == 0) { leading_zeros += 16; n <<= 16; }
    if ((n & 0xff000000) == 0) { leading_zeros += 8; n <<= 8; }
    if ((n & 0xf0000000) == 0) { leading_zeros += 4; n <<= 4; }
    if ((n & 0xc0000000) == 0) { leading_zeros += 2; n <<= 2; }
    if ((n & 0x80000000) == 0) { leading_zeros += 1; }
    return 32 - leading_zeros;
}
