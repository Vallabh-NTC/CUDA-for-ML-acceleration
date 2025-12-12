#pragma once
#include <cstdint>

struct SARA_10_Data
{
    float accel_x = 0.0f;
    float accel_y = 0.0f;
    float omega_z = 0.0f;
};

class SARA_10
{
public:
    void decode(const unsigned char* pdus);   // base PDU pointer
    const SARA_10_Data& data() const { return d_; }

private:
    SARA_10_Data d_{};
};
