#pragma once
#include <cstdint>

struct Motor14Data
{
    uint8_t mo_bls = 0;
};

class Motor14
{
public:
    void decode(const unsigned char* data);
    const Motor14Data& data() const { return out_; }

private:
    Motor14Data out_{};
};
