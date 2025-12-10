#include "ESP21.hpp"
#include "msg_common.hpp"   // your bit reader

void ESP21::decode(const unsigned char* buff)
{
    // esp_v_signal: byte 4, bit 0, len = 16
    uint16_t raw = compute_signal_raw_value<uint16_t>(buff, 4, 0, 16);

    d.valid = (raw <= 65532);

    if (d.valid)
        d.vehicle_speed = raw * 0.01f;  // m/s
    else
        d.vehicle_speed = NAN;
}
