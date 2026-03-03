#pragma once
#include <cstdint>

struct SARA_06_Data
{
    float accel_x = 0.0f;
    float accel_y = 0.0f;
    float omega_z = 0.0f;
};

class SARA_06
{
public:
    void decode(const unsigned char* pdus);
    const SARA_06_Data& data() const { return d_; }

private:
    SARA_06_Data d_{};
};
