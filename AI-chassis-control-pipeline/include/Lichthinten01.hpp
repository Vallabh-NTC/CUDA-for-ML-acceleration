#pragma once
#include <cstdint>

class Lichthinten01 {
public:
    uint8_t raw[7] = {0};  // 7 bytes message
    void decode(const unsigned char* data);

    // Build a bitmask summarizing all states
    uint32_t compute_mask() const;

    void print() const;
};
