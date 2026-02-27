#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "AdmaProtocolPackets.hpp"

namespace adma {

enum class ProtocolVersion {
  V32,
  V333,
  V334,
  V335,
};

struct DecodedKinematics {
  double accx_g;
  double accy_g;
  double accz_g;
  double speed_kmh;
};

struct DecodedPacket {
  ProtocolVersion version;
  DecodedKinematics kinematics;
  std::optional<protocol::AdmaDataV32> v32;
  std::optional<protocol::AdmaDataV333> v333;
  std::optional<protocol::AdmaDataV334> v334;
  std::optional<protocol::AdmaDataV335> v335;
};

class AdmaPacketDecoder {
 public:
  explicit AdmaPacketDecoder(ProtocolVersion version);

  DecodedPacket decode(const std::vector<std::byte>& payload) const;

  static ProtocolVersion parseVersion(std::string_view value);
  static std::string toString(ProtocolVersion version);
  static std::size_t expectedPacketSize(ProtocolVersion version);

 private:
  ProtocolVersion version_;

  static double scale(int32_t raw, double factor);
  static DecodedKinematics fromV32(const protocol::AdmaDataV32& packet);
  static DecodedKinematics fromV333(const protocol::AdmaDataV333& packet);
  static DecodedKinematics fromV334(const protocol::AdmaDataV334& packet);
  static DecodedKinematics fromV335(const protocol::AdmaDataV335& packet);
};

}  // namespace adma
