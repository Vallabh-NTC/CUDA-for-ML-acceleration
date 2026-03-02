#include "LWI01.hpp"
#include "BitExtractor.hpp"
#include <iostream>
#include <cmath>

void LWI01::decode(const unsigned char* data)
{
    crc = BitExtractor::extract(data, 0, 0, 8);

    uint16_t angle_raw = BitExtractor::extract(data, 2, 0, 13);
    angle = (angle_raw <= 8000) ? angle_raw * 0.1f : NAN;

    angle_sign = static_cast<uint8_t>(BitExtractor::extract(data, 3, 5, 1));
    if (angle_sign)
        angle = -angle;

    uint16_t speed_raw = BitExtractor::extract(data, 3, 7, 9);
    speed = (speed_raw <= 500) ? speed_raw * 5.0f : NAN;

    speed_sign = static_cast<uint8_t>(BitExtractor::extract(data, 3, 6, 1));
    if (!std::isnan(speed) && speed_sign)
        speed = -speed;
}

void LWI01::print() const
{
    std::cout << "[LWI01] angle=" << angle
              << " speed=" << speed
              << " crc=" << int(crc) << "\n";
}
