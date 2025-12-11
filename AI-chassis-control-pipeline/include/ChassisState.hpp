#pragma once
#include <cstdint>

struct ChassisState
{
    float steering_angle = 0.f;
    float steering_speed = 0.f;
    uint32_t lights_rear = 0;
    uint64_t timestamp_ms = 0;
};
