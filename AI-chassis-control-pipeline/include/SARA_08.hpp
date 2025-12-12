#pragma once
#include <cstdint>

struct SARA_08_Data
{
    float omega_x = 0.0f;
    float omega_y = 0.0f;
    float accel_z = 0.0f;
};

class SARA_08
{
public:
    void decode(const unsigned char* pdus);   // base PDU pointer
    const SARA_08_Data& data() const { return d_; }

private:
    SARA_08_Data d_{};
};
