#pragma once
#include <winsock2.h>
#include <cstdint>
#include <vector>

class UdpReceiver {
public:
    UdpReceiver(uint16_t port);
    ~UdpReceiver();

    int receive(unsigned char* buffer, size_t size);

private:
    SOCKET sock;
};
