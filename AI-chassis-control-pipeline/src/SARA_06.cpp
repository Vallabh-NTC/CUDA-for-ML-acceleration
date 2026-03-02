#include "SARA_06.hpp"
#include "BitExtractor.hpp"
#include <cmath>

void SARA_06::decode(const unsigned char* pdus)
{
    const uint16_t ax_raw = BitExtractor::extract(pdus, 1, 4, 16);
    const uint16_t ay_raw = BitExtractor::extract(pdus, 3, 4, 16);
    const uint16_t oz_raw = BitExtractor::extract(pdus, 5, 4, 16);

    d_.accel_x = (ax_raw >= 1 && ax_raw <= 65534) ? (ax_raw * 0.02f - 655.36f) : NAN;
    d_.accel_y = (ay_raw >= 1 && ay_raw <= 65534) ? (ay_raw * 0.02f - 655.36f) : NAN;
    d_.omega_z = (oz_raw >= 1 && oz_raw <= 65534) ? (oz_raw * 0.01f - 327.68f) : NAN;
}
