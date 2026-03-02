#pragma once
#include <cstdint>

struct ESP05Data
{
    float brake_pressure = 0.0f;
};

class ESP05
{
public:
    void decode(const unsigned char* data);
    const ESP05Data& data() const { return out_; }

private:
    ESP05Data out_{};
};
