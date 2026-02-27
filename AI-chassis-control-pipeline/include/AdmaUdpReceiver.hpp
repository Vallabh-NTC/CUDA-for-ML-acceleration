#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace adma {

class AdmaUdpReceiver {
 public:
  AdmaUdpReceiver(const std::string& listen_ip, uint16_t listen_port, std::optional<std::string> expected_source_ip);
  ~AdmaUdpReceiver();

  AdmaUdpReceiver(const AdmaUdpReceiver&) = delete;
  AdmaUdpReceiver& operator=(const AdmaUdpReceiver&) = delete;

  std::vector<std::byte> receive();

 private:
#ifdef _WIN32
  using SocketHandle = uintptr_t;
  static constexpr SocketHandle kInvalidSocket = static_cast<SocketHandle>(~0ULL);
#else
  using SocketHandle = int;
  static constexpr SocketHandle kInvalidSocket = -1;
#endif

  SocketHandle socket_;
  std::optional<std::string> expected_source_ip_;

  void closeSocket() noexcept;
};

}  // namespace adma
