#pragma once
#include <cstdint>

//
// Minimal Motor20 decoder
// Extracts: gas pedal percentage (0–100%)
//

// Struct holding the decoded values
struct Motor20Data
{
    float gas_percent = 0.0f;   // 0–100%
    bool valid = false;         // from phy_valid
};

class Motor20
{
public:
    Motor20() = default;

    // Decode a PDU starting at buff (pdus + 629)
    void decode(const unsigned char* buff);

    // Get latest decoded data
    const Motor20Data& data() const { return out; }

private:
    Motor20Data out;

    // Extract a raw value from "buff" at byte offset, bit offset, bit length
    template<typename T>
    static T extract(const unsigned char* buff, int byte, int bit, int length);
};
