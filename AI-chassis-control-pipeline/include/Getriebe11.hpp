#pragma once
#include <cstdint>

struct Getriebe11Data
{
    uint8_t ge_zielgang = 0;
};

class Getriebe11
{
public:
    void decode(const unsigned char* data);
    const Getriebe11Data& data() const { return out_; }

private:
    Getriebe11Data out_{};
};
