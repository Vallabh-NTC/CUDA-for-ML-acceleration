#pragma once
#include <cstdint>

class BitExtractor {
public:
    static uint64_t extract(const unsigned char* buf, int byte, int bit, int len);
};
