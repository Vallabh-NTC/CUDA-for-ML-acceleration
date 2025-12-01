#pragma once
#include <cstdint>
#include <cmath>

class BrakeEV01 {
public:
    float brake_percent = NAN;
    void decode(const unsigned char* data);
};
