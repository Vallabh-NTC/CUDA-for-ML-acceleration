#pragma once
#include <string>
#include <cstdint>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

class UdpSender {
public:
    UdpSender(const std::string& ip, uint16_t port);
    ~UdpSender();

    bool send(const std::string& msg);

private:
    int sock = -1;
    struct sockaddr_in addr {};
};
