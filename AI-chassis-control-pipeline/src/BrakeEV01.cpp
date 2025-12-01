#include "BrakeEV01.hpp"
#include "BitExtractor.hpp"

void BrakeEV01::decode(const unsigned char* data)
{
    uint8_t raw = BitExtractor::extract(data, 3, 0, 8);

    if (raw >= 1 && raw <= 254)
        brake_percent = raw * 0.4f - 0.4f;
    else
        brake_percent = NAN;
}
