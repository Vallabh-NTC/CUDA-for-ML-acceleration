#pragma once
#include <cstdint>
#include <cmath>

class BrakeEV01 {
public:
    float brake_percent = NAN;
    float pedal_position = NAN;
    uint8_t driver_brakes = 0;
    void decode(const unsigned char* data);
};
