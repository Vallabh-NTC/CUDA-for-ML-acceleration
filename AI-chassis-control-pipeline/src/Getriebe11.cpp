#include "Getriebe11.hpp"
#include "BitExtractor.hpp"

void Getriebe11::decode(const unsigned char* data)
{
    out_.ge_zielgang = static_cast<uint8_t>(BitExtractor::extract(data, 7, 4, 4));
}
