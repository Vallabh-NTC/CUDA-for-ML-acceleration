#include "KlimaSensor02.hpp"
#include "BitExtractor.hpp"
#include <cmath>

void KlimaSensor02::decode(const unsigned char* data)
{
    const uint8_t raw = static_cast<uint8_t>(BitExtractor::extract(data, 0, 0, 8));
    out_.external_temperature = (raw <= 252) ? (raw * 0.5f - 50.0f) : NAN;
}
