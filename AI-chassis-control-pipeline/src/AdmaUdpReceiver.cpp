#include "AdmaUdpReceiver.hpp"

#include <array>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace adma {

#ifdef _WIN32
class WinsockGuard {
 public:
  WinsockGuard() {
    WSADATA wsa_data{};
    const int result = WSAStartup(MAKEWORD(2, 2), &wsa_data);
    if (result != 0) {
      throw std::runtime_error("WSAStartup failed with error: " + std::to_string(result));
    }
  }

  ~WinsockGuard() { WSACleanup(); }
};
#endif

AdmaUdpReceiver::AdmaUdpReceiver(
    const std::string& listen_ip,
    uint16_t listen_port,
    std::optional<std::string> expected_source_ip)
    : socket_(kInvalidSocket), expected_source_ip_(std::move(expected_source_ip)) {
#ifdef _WIN32
  static WinsockGuard winsock_guard;
#endif

  socket_ = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (socket_ == kInvalidSocket) {
    throw std::runtime_error("Unable to create UDP socket");
  }

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(listen_port);

  const int pton_result = inet_pton(AF_INET, listen_ip.c_str(), &address.sin_addr);
  if (pton_result != 1) {
    closeSocket();
    throw std::runtime_error("Invalid listen IP: " + listen_ip);
  }

  if (::bind(socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    closeSocket();
    throw std::runtime_error("Failed to bind UDP socket on " + listen_ip + ":" + std::to_string(listen_port));
  }
}

AdmaUdpReceiver::~AdmaUdpReceiver() { closeSocket(); }

std::vector<std::byte> AdmaUdpReceiver::receive() {
  std::array<std::byte, 2048> buffer{};

  while (true) {
    sockaddr_in source{};
#ifdef _WIN32
    int source_len = sizeof(source);
#else
    socklen_t source_len = sizeof(source);
#endif

    const int bytes = ::recvfrom(
        socket_, reinterpret_cast<char*>(buffer.data()), static_cast<int>(buffer.size()), 0,
        reinterpret_cast<sockaddr*>(&source), &source_len);

    if (bytes <= 0) {
      throw std::runtime_error("recvfrom failed");
    }

    if (expected_source_ip_.has_value()) {
      char source_ip_str[INET_ADDRSTRLEN]{};
      const char* ntop_result = inet_ntop(AF_INET, &source.sin_addr, source_ip_str, INET_ADDRSTRLEN);
      if (ntop_result == nullptr) {
        continue;
      }
      if (expected_source_ip_.value() != std::string(source_ip_str)) {
        continue;
      }
    }

    return std::vector<std::byte>(buffer.begin(), buffer.begin() + bytes);
  }
}

void AdmaUdpReceiver::closeSocket() noexcept {
  if (socket_ == kInvalidSocket) {
    return;
  }
#ifdef _WIN32
  closesocket(socket_);
#else
  close(socket_);
#endif
  socket_ = kInvalidSocket;
}

}  // namespace adma
