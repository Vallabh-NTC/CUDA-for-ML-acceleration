#pragma once
#include <cstdint>

struct ESP03Data
{
    float wheel_speed_fl = 0.0f;
    float wheel_speed_fr = 0.0f;
    float wheel_speed_rl = 0.0f;
    float wheel_speed_rr = 0.0f;
};

class ESP03
{
public:
    void decode(const unsigned char* data);
    const ESP03Data& data() const { return out_; }

private:
    ESP03Data out_{};
};
