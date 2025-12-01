#pragma once
#include <cstddef>
#include <cstdint>

class UdpReceiver {
public:
    explicit UdpReceiver(uint16_t port);
    ~UdpReceiver();

    int receive(unsigned char* buffer, size_t size);

private:
    int sock = -1;
};
