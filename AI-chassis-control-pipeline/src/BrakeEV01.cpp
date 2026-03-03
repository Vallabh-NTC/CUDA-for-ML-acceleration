#include "BrakeEV01.hpp"
#include "BitExtractor.hpp"

void BrakeEV01::decode(const unsigned char* data)
{
    uint8_t raw = BitExtractor::extract(data, 3, 0, 8);

    if (raw >= 1 && raw <= 254) {
        pedal_position = raw * 0.4f - 0.4f;
        brake_percent = pedal_position;
    } else {
        pedal_position = NAN;
        brake_percent = NAN;
    }

    driver_brakes = static_cast<uint8_t>(BitExtractor::extract(data, 4, 4, 1));
}
