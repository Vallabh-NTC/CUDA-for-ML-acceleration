#pragma once
#include <cstdint>

struct LHEPS03Data
{
    float steering_torque = 0.0f;
    uint8_t steering_torque_sign = 0;
};

class LHEPS03
{
public:
    void decode(const unsigned char* data);
    const LHEPS03Data& data() const { return out_; }

private:
    LHEPS03Data out_{};
};
