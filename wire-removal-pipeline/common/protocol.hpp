#pragma once
#include <cstdint>

// Build a little-endian FOURCC at compile time.
// Works in C++11+ (no memcpy / no std::string needed).
constexpr uint32_t make_fourcc_le(char a, char b, char c, char d) {
    return (static_cast<uint32_t>(static_cast<unsigned char>(a))      ) |
           (static_cast<uint32_t>(static_cast<unsigned char>(b)) <<  8) |
           (static_cast<uint32_t>(static_cast<unsigned char>(c)) << 16) |
           (static_cast<uint32_t>(static_cast<unsigned char>(d)) << 24);
}

// ---- headers on the wire ----------------------------------------------------
// FrameHeader: uint32 magic; uint32 width; uint32 height; uint32 stride;
//              uint32 channels; uint32 size;
struct FrameHeader {
    uint32_t magic;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t channels;
    uint32_t size;
};

// MaskHeader: uint32 magic; uint32 width; uint32 height; uint32 size;
// (size must be width*height bytes)
struct MaskHeader {
    uint32_t magic;
    uint32_t width;
    uint32_t height;
    uint32_t size;
};

// New: MasksCountHeader: uint32 magic; uint32 count;
struct MasksCountHeader {
    uint32_t magic;
    uint32_t count;
};

// Magic constants (little-endian)
static constexpr uint32_t FRAM_MAGIC = make_fourcc_le('F','R','A','M');
static constexpr uint32_t MASK_MAGIC = make_fourcc_le('M','A','S','K');
static constexpr uint32_t MCNT_MAGIC = make_fourcc_le('M','C','N','T');
