#include "UdpReceiver.hpp"
#include <ws2tcpip.h>
#include <iostream>

#pragma comment(lib, "Ws2_32.lib")

UdpReceiver::UdpReceiver(uint16_t port) {
    WSADATA wsa;
    WSAStartup(MAKEWORD(2,2), &wsa);

    sock = socket(AF_INET, SOCK_DGRAM, 0);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    bind(sock, (sockaddr*)&addr, sizeof(addr));
}

UdpReceiver::~UdpReceiver() {
    closesocket(sock);
    WSACleanup();
}

int UdpReceiver::receive(unsigned char* buffer, size_t size) {
    return recv(sock, (char*)buffer, size, 0);
}
