#include "ESP21.hpp"
#include "BitExtractor.hpp"
#include <cmath> 

void ESP21::decode(const unsigned char* data)
{
    // extract 16-bit value starting at byte 4, bit 0
    uint32_t raw = BitExtractor::extract(data, 4, 0, 16);

    d.valid = (raw <= 65532);

    if (d.valid)
        d.vehicle_speed = raw * 0.01f;   // m/s
    else
        d.vehicle_speed = NAN;
}
