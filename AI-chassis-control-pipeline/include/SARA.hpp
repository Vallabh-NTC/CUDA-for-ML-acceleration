#pragma once
#include <cstdint>
#include <string>

struct SARA07Data {
    float nickwinkel = 0.f;   // pitch
    float wankwinkel = 0.f;   // roll
};

struct SARA08Data {
    float omega_x = 0.f;
    float omega_y = 0.f;
    float accel_z = 0.f;
};

struct SARA10Data {
    float accel_x = 0.f;
    float accel_y = 0.f;
    float omega_z = 0.f;
};

class SARA
{
public:
    SARA07Data d07;
    SARA08Data d08;
    SARA10Data d10;

    void decode_all(const unsigned char* pdus);

    // Produces CSV: ax,ay,az,ox,oy,oz,nick,wank
    std::string to_csv() const;
};
