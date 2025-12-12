#include "SARA_10.hpp"
#include "BitExtractor.hpp"

void SARA_10::decode(const unsigned char* pdus)
{
    const unsigned char* b10 = pdus;

    uint16_t ax_raw = BitExtractor::extract(b10, 1, 4, 16);
    uint16_t ay_raw = BitExtractor::extract(b10, 3, 4, 16);
    uint16_t oz_raw = BitExtractor::extract(b10, 5, 4, 16);

    d_.accel_x = (ax_raw * 0.02f) - 655.36f;
    d_.accel_y = (ay_raw * 0.02f) - 655.36f;
    d_.omega_z = (oz_raw * 0.01f) - 327.68f;
}
