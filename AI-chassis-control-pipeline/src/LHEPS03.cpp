#include "LHEPS03.hpp"
#include "BitExtractor.hpp"
#include <cmath>

void LHEPS03::decode(const unsigned char* data)
{
    const uint16_t raw = BitExtractor::extract(data, 5, 0, 10);
    out_.steering_torque = (raw <= 800) ? (raw * 0.01f) : NAN;
    out_.steering_torque_sign = static_cast<uint8_t>(BitExtractor::extract(data, 6, 7, 1));
}
