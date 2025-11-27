#include "LWI01.hpp"

void LWI01::decode(const unsigned char* data) {
    crc = BitExtractor::extract(data, 0, 0, 8);
    
    uint16_t angle_raw = BitExtractor::extract(data, 2, 0, 13);
    bool valid_angle = angle_raw <= 8000;
    angle = valid_angle ? angle_raw * 0.1f : NAN;

    uint8_t sign_bit = BitExtractor::extract(data, 3, 5, 1);
    if(sign_bit) angle = -angle;

    uint16_t speed_raw = BitExtractor::extract(data, 3, 7, 9);
    bool valid_speed = speed_raw <= 500;
    speed = valid_speed ? speed_raw * 5.0f : NAN;
}

void LWI01::print() const {
    std::cout << "CRC=" << int(crc)
              << " | Angle=" << angle
              << " deg | Speed=" << speed << " deg/s\n";
}
