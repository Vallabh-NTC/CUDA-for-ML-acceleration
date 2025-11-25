#include <iostream>
#include <cstring>
#include <arpa/inet.h>
#include <unistd.h>
#include <cmath>

// ----------------------- Bit Extractor -------------------------
static uint64_t extract_bits(const unsigned char *buf, int byte, int bit, int len)
{
    uint64_t val = 0;
    int bitpos = byte * 8 + bit;

    for(int i = 0; i < len; ++i)
    {
        int b = bitpos + i;
        int srcByte = b / 8;
        int srcBit  = b % 8;

        uint8_t bitval = (buf[srcByte] >> srcBit) & 1;
        val |= (uint64_t)bitval << i;
    }
    return val;
}

int main()
{
    int sock = socket(AF_INET, SOCK_DGRAM, 0);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(1500);
    addr.sin_addr.s_addr = INADDR_ANY;

    bind(sock, (sockaddr*)&addr, sizeof(addr));
    std::cout << "Listening on Jetson (UDP 1500)…\n";

    unsigned char buf[4096];

    while(true)
    {
        int n = recv(sock, buf, sizeof(buf), 0);
        if (n <= 0) continue;
        if (n < 700) continue;

        int cluster = buf[0];
        if (cluster != 1) continue;

        const unsigned char *pdus = buf + 3;

        // 🔥 Your steering PDU offset
        const unsigned char *lwi = pdus + 677;

        uint8_t crc = extract_bits(lwi, 0, 0, 8);

        uint16_t raw_angle = extract_bits(lwi, 2, 0, 13);
        bool angle_valid = raw_angle <= 8000;
        float angle = angle_valid ? raw_angle * 0.1f : NAN;

        uint8_t sign_bit = extract_bits(lwi, 3, 5, 1);
        if (sign_bit) angle = -angle;

        uint16_t raw_speed = extract_bits(lwi, 3, 7, 9);
        bool speed_valid = raw_speed <= 500;
        float speed = speed_valid ? raw_speed * 5.0f : NAN;

        std::cout << "CRC=" << int(crc)
                  << " | Angle=" << angle
                  << " | Speed=" << speed << "\n";
    }

    close(sock);
    return 0;
}
