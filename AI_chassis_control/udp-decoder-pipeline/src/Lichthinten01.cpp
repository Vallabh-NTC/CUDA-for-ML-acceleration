#include "Lichthinten01.hpp"

void Lichthinten01::decode(const unsigned char* data) {
    bremslicht_h_aktiv = BitExtractor::extract(data, 1, 3, 1);
    rueckfahrlicht_aktiv = BitExtractor::extract(data, 1, 5, 1);
    blinker_hl = BitExtractor::extract(data, 1, 6, 1);
    blinker_hr = BitExtractor::extract(data, 1, 7, 1);
}

void Lichthinten01::print() const {
    std::cout << "--- LICHTHINTEN01 ---\n";
    std::cout << "Bremslicht hinten aktiv: " << int(bremslicht_h_aktiv) << "\n";
    std::cout << "Rueckfahrlicht aktiv:    " << int(rueckfahrlicht_aktiv) << "\n";
    std::cout << "Blinker HL:              " << int(blinker_hl) << "\n";
    std::cout << "Blinker HR:              " << int(blinker_hr) << "\n";
}
