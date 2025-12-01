#include "UdpSender.hpp"
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <iostream>

UdpSender::UdpSender(const std::string& ip, uint16_t port)
{
    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("UdpSender socket()");
        return;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);

    if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) <= 0) {
        std::cerr << "UdpSender inet_pton failed for " << ip << "\n";
    }

    std::cout << "[UdpSender] Ready to send to "
              << ip << ":" << port << "\n";
}

UdpSender::~UdpSender()
{
    if (sock >= 0)
        close(sock);
}

bool UdpSender::send(const std::string& msg)
{
    if (sock < 0) return false;

    int n = sendto(sock, msg.c_str(), msg.size(), 0,
                   (sockaddr*)&addr, sizeof(addr));

    return (n == (int)msg.size());
}
