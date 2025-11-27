#pragma once
#include "PDU.hpp"
#include "BitExtractor.hpp"
#include <cmath>
#include <iostream>

class LWI01 : public PDU {
public:
    uint8_t crc;
    float angle;
    float speed;

    void decode(const unsigned char* data) override;
    void print() const override;
};
