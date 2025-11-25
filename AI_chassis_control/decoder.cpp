#include <iostream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <cstdint>
#include <cmath>

#pragma comment(lib, "Ws2_32.lib")

// ----------------------- Bit Extractor -------------------------
static uint64_t extract_bits(const unsigned char *buf, int byte, int bit, int len)
{
    uint64_t val = 0;
    int bitpos = byte * 8 + bit;

    for(int i = 0; i < len; i++)
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
    WSADATA wsa;
    WSAStartup(MAKEWORD(2,2), &wsa);

    SOCKET sock = socket(AF_INET, SOCK_DGRAM, 0);

    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(1500);

    bind(sock, (sockaddr *)&addr, sizeof(addr));
    std::cout << "Listening for UDP packets on port 1500...\n";

    unsigned char buf[4096];

    while(true)
    {
        int n = recv(sock, (char*)buf, sizeof(buf), 0);
        if(n <= 0) continue;

        if(n < 700)
        {
            std::cout << "Packet too small\n";
            continue;
        }

        int cluster = buf[0];
        if(cluster != 1)
            continue;

        const unsigned char *pdus = buf + 3;
        const unsigned char *lwi  = pdus + 677;     // LWI01 offset = 680

        uint8_t crc = extract_bits(lwi, 0, 0, 8);

        uint16_t angle_raw = extract_bits(lwi, 2, 0, 13);
        bool valid_angle = angle_raw <= 8000;
        float angle = valid_angle ? angle_raw * 0.1f : NAN;

        uint8_t sign_bit = extract_bits(lwi, 3, 5, 1);
        if(sign_bit) angle = -angle;

        uint16_t speed_raw = extract_bits(lwi, 3, 7, 9);
        bool valid_speed = speed_raw <= 500;
        float speed = valid_speed ? speed_raw * 5.0f : NAN;

        std::cout << "CRC=" << int(crc)
                  << " | Angle=" << angle
                  << " deg | Speed=" << speed << " deg/s\n";
    }

    closesocket(sock);
    WSACleanup();
    return 0;
}
