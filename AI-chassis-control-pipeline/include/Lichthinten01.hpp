#pragma once
#include "PDU.hpp"
#include "BitExtractor.hpp"
#include <iostream>

class Lichthinten01 : public PDU {
public:
    uint8_t bremslicht_h_aktiv;
    uint8_t rueckfahrlicht_aktiv;
    uint8_t blinker_hl;
    uint8_t blinker_hr;

    void decode(const unsigned char* data) override;
    void print() const override;
};
