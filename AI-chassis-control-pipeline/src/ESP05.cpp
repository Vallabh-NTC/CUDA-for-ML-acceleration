#include "ESP05.hpp"
#include "BitExtractor.hpp"
#include <cmath>

void ESP05::decode(const unsigned char* data)
{
    const uint16_t raw = BitExtractor::extract(data, 2, 0, 10);
    out_.brake_pressure = (raw <= 1022) ? (raw * 0.3f - 30.0f) : NAN;
}
