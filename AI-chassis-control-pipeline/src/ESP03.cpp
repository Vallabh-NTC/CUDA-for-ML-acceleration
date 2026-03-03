#include "ESP03.hpp"
#include "BitExtractor.hpp"
#include <cmath>

void ESP03::decode(const unsigned char* data)
{
    const uint16_t fl_raw = BitExtractor::extract(data, 2, 0, 12);
    const uint16_t fr_raw = BitExtractor::extract(data, 3, 4, 12);
    const uint16_t rl_raw = BitExtractor::extract(data, 5, 0, 12);
    const uint16_t rr_raw = BitExtractor::extract(data, 6, 4, 12);

    out_.wheel_speed_fl = (fl_raw <= 4092) ? (fl_raw * 0.1f) : NAN;
    out_.wheel_speed_fr = (fr_raw <= 4092) ? (fr_raw * 0.1f) : NAN;
    out_.wheel_speed_rl = (rl_raw <= 4092) ? (rl_raw * 0.1f) : NAN;
    out_.wheel_speed_rr = (rr_raw <= 4092) ? (rr_raw * 0.1f) : NAN;
}
