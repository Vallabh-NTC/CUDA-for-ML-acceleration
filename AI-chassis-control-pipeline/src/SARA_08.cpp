#include "SARA_08.hpp"
#include "BitExtractor.hpp"

void SARA_08::decode(const unsigned char* pdus)
{
    const unsigned char* b08 = pdus;

    uint16_t ox_raw = BitExtractor::extract(b08, 1, 4, 16);
    uint16_t oy_raw = BitExtractor::extract(b08, 3, 4, 16);
    uint16_t az_raw = BitExtractor::extract(b08, 5, 4, 16);

    d_.omega_x = (ox_raw * 0.01f) - 327.68f;
    d_.omega_y = (oy_raw * 0.01f) - 327.68f;
    d_.accel_z = (az_raw * 0.02f) - 655.36f;
}
