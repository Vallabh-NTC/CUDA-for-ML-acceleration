#pragma once
#include <cstdint>

struct KlimaSensor02Data
{
    float external_temperature = 0.0f;
};

class KlimaSensor02
{
public:
    void decode(const unsigned char* data);
    const KlimaSensor02Data& data() const { return out_; }

private:
    KlimaSensor02Data out_{};
};
