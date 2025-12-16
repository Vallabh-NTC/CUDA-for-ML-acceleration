#include "Motor20.hpp"
#include <cmath>
#include <cassert>
#include <algorithm>
#include <cstdint>

//
// Local copy of Jupiter's compute_signal_raw_value
// (slightly simplified, but bit-identical)
//
template <typename T>
static T compute_signal_raw_value_local(
    const unsigned char* pdu,
    unsigned int start_byte,
    unsigned int start_bit,
    unsigned int length_bit)
{
    assert(start_bit <= 7);
    assert(length_bit <= sizeof(T) * 8);

    int length = length_bit;
    T sig = 0;
    int dest_bit = 0;

    while (length > 0) {
        uint8_t mask = (1u << std::min(8, length)) - 1u;
        sig |= static_cast<T>(
                   ((pdu[start_byte] >> start_bit) & mask)
               ) << dest_bit;

        int consumed = 8 - start_bit;
        length     -= consumed;
        dest_bit   += consumed;
        start_bit   = 0;
        start_byte += 1;
    }

    // Sign-extension (noop for unsigned)
    int upper_bits = sizeof(T) * 8 - length_bit;
    return static_cast<T>(sig << upper_bits) >> upper_bits;
}

//
// Decode Motor20 PDU (Jupiter-equivalent)
//
void Motor20::decode(const unsigned char* buff)
{
    Motor20Data result{};

    // mo_fahrpedalrohwert_01
    // start_byte = 1
    // start_bit  = 4
    // length     = 8
    uint8_t raw =
        compute_signal_raw_value_local<uint8_t>(buff, 1, 4, 8);

    if (raw <= 254) {
        result.valid = true;
        result.gas_percent = raw * 0.4f;
    } else {
        result.valid = false;
        result.gas_percent = NAN;
    }

    out = result;
}
