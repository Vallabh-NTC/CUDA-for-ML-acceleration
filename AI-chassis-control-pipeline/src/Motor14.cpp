#include "Motor14.hpp"
#include "BitExtractor.hpp"

void Motor14::decode(const unsigned char* data)
{
    out_.mo_bls = static_cast<uint8_t>(BitExtractor::extract(data, 3, 6, 1));
}
