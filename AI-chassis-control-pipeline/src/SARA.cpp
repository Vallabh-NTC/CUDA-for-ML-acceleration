#include "SARA.hpp"
#include "BitExtractor.hpp"   // your existing bit extractor
#include <sstream>
#include <iomanip>
#include <cmath>

void SARA::decode_all(const unsigned char* pdus)
{
    // ------------------ SARA10 (offset +412) ------------------
    const unsigned char* b10 = pdus + 412;

    uint16_t ax_raw = BitExtractor::extract(b10, 1, 4, 16);
    uint16_t ay_raw = BitExtractor::extract(b10, 3, 4, 16);
    uint16_t oz_raw = BitExtractor::extract(b10, 5, 4, 16);

    d10.accel_x = (ax_raw * 0.02f) - 655.36f;
    d10.accel_y = (ay_raw * 0.02f) - 655.36f;
    d10.omega_z = (oz_raw * 0.01f) - 327.68f;

    // ------------------ SARA08 (offset +437) ------------------
    const unsigned char* b08 = pdus + 437;

    uint16_t ox_raw = BitExtractor::extract(b08, 1, 4, 16);
    uint16_t oy_raw = BitExtractor::extract(b08, 3, 4, 16);
    uint16_t az_raw = BitExtractor::extract(b08, 5, 4, 16);

    d08.omega_x = (ox_raw * 0.01f) - 327.68f;
    d08.omega_y = (oy_raw * 0.01f) - 327.68f;
    d08.accel_z = (az_raw * 0.02f) - 655.36f;

    // ------------------ SARA07 (offset +454) ------------------
    const unsigned char* b07 = pdus + 454;

    uint16_t nick_raw = BitExtractor::extract(b07, 1, 6, 13);
    uint16_t wank_raw = BitExtractor::extract(b07, 3, 3, 13);

    d07.nickwinkel = (nick_raw * 0.025f) - 102.3f;
    d07.wankwinkel = (wank_raw * 0.025f) - 102.3f;
}

std::string SARA::to_csv() const
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3)
       << d10.accel_x << "," << d10.accel_y << "," << d08.accel_z << ","
       << d08.omega_x << "," << d08.omega_y << "," << d10.omega_z << ","
       << d07.nickwinkel << "," << d07.wankwinkel;

    return ss.str();
}
