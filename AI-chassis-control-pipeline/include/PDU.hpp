#pragma once
#include <cstdint>

class PDU {
public:
    virtual void decode(const unsigned char* data) = 0;
    virtual void print() const = 0;
    virtual ~PDU() = default;
};
