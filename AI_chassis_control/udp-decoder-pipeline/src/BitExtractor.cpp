#include "BitExtractor.hpp"

uint64_t BitExtractor::extract(const unsigned char* buf, int byte, int bit, int len) {
    uint64_t val = 0;
    int bitpos = byte * 8 + bit;

    for(int i = 0; i < len; i++) {
        int b = bitpos + i;
        int srcByte = b / 8;
        int srcBit  = b % 8;
        uint8_t bitval = (buf[srcByte] >> srcBit) & 1;
        val |= (uint64_t)bitval << i;
    }
    return val;
}
