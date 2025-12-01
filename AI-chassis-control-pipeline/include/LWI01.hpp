#pragma once
#include <cstdint>

class LWI01 {
public:
    uint8_t crc = 0;
    float angle = 0;
    float speed = 0;

    void decode(const unsigned char* data);
    void print() const;
};
