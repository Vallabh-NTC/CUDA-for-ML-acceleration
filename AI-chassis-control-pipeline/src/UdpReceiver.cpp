#include "UdpReceiver.hpp"
#include <iostream>
#include <cstring>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>

UdpReceiver::UdpReceiver(uint16_t port)
{
    // -------------------------------
    // Create UDP Socket
    // -------------------------------
    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("socket() failed");
        return;
    }

    // -------------------------------
    // Allow reusing address/port
    // -------------------------------
    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));

    // -------------------------------
    // Bind to explicit Xavier IP first
    // (replace with your actual IP!)
    // -------------------------------
    const char* xavier_ip = "192.168.1.197";

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);

    if (inet_pton(AF_INET, xavier_ip, &addr.sin_addr) <= 0) {
        std::cerr << "inet_pton failed for " << xavier_ip
                  << " — falling back to INADDR_ANY\n";
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    }

    if (bind(sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind() failed");
    } else {
        std::cout << "[UdpReceiver] Bound to " << xavier_ip
                  << ":" << port << "\n";
    }

    // -------------------------------
    // Add timeout so recv() does not block forever
    // -------------------------------
    struct timeval tv;
    tv.tv_sec = 1;      // 1 sec timeout
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}


UdpReceiver::~UdpReceiver()
{
    close(sock);
}


int UdpReceiver::receive(unsigned char* buffer, size_t size)
{
    int n = recv(sock, buffer, size, 0);

    if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // Timeout: nothing received
            return 0;
        }
        perror("recv() error");
        return -1;
    }

    return n;
}
