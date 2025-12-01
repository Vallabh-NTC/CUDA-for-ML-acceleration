#include "Lichthinten01.hpp"
#include "BitExtractor.hpp"
#include <iostream>

void Lichthinten01::decode(const unsigned char* data)
{
    for (int i = 0; i < 7; i++)
        raw[i] = data[i];
}

uint32_t Lichthinten01::compute_mask() const
{
    uint32_t m = 0;
    for (int i = 0; i < 7; i++)
        m |= (uint32_t(raw[i]) << (i * 8));
    return m;
}

void Lichthinten01::print() const
{
    std::cout << "[LH] mask=0x" << std::hex << compute_mask() << std::dec << "\n";
}
