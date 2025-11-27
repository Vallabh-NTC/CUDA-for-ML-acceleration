#pragma once
#include <cstdint>
 
#ifdef _WIN32
    #include <winsock2.h>
#else
    #include <sys/socket.h>
    #include <arpa/inet.h>
#endif

class UdpReceiver {
public:
    UdpReceiver(uint16_t port);
    ~UdpReceiver();

    int receive(unsigned char* buffer, size_t size);

private:
    #ifdef _WIN32
        SOCKET sock;
    #else
        int sock;
    #endif
};
