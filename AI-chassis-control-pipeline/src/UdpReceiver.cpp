#include "UdpReceiver.hpp"
#include <iostream>
 
#ifdef _WIN32
    #include <ws2tcpip.h>
    #pragma comment(lib, "Ws2_32.lib")
#else
    #include <unistd.h>
#endif

UdpReceiver::UdpReceiver(uint16_t port) {
   #ifdef _WIN32

    WSADATA wsa;

    WSAStartup(MAKEWORD(2,2), &wsa);

    sock = socket(AF_INET, SOCK_DGRAM, 0);

#else

    sock = socket(AF_INET, SOCK_DGRAM, 0);

#endif
 
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    bind(sock, (sockaddr*)&addr, sizeof(addr));
}

UdpReceiver::~UdpReceiver() {
    #ifdef _WIN32
        closesocket(sock);
        WSACleanup();
    #else
        close(sock);
    #endif
}

int UdpReceiver::receive(unsigned char* buffer, size_t size) {
    return recv(sock, (char*)buffer, size, 0);
}
