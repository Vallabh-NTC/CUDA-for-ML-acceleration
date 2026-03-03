#include "AdmaDecoder.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace adma {

namespace {

double speedKmhFromMs(double vx_ms, double vy_ms) {
  return std::sqrt(vx_ms * vx_ms + vy_ms * vy_ms) * 3.6;
}

}  // namespace

AdmaPacketDecoder::AdmaPacketDecoder(ProtocolVersion version) : version_(version) {}

DecodedPacket AdmaPacketDecoder::decode(const std::vector<std::byte>& payload) const {
  const auto expected_size = expectedPacketSize(version_);
  if (payload.size() != expected_size) {
    throw std::runtime_error(
        "Invalid payload size for protocol " + toString(version_) +
        ": got " + std::to_string(payload.size()) +
        ", expected " + std::to_string(expected_size));
  }

  DecodedPacket out{};
  out.version = version_;

  switch (version_) {
    case ProtocolVersion::V32: {
      protocol::AdmaDataV32 packet{};
      std::memcpy(&packet, payload.data(), sizeof(packet));
      out.kinematics = fromV32(packet);
      out.v32 = packet;
      return out;
    }
    case ProtocolVersion::V333: {
      protocol::AdmaDataV333 packet{};
      std::memcpy(&packet, payload.data(), sizeof(packet));
      out.kinematics = fromV333(packet);
      out.v333 = packet;
      return out;
    }
    case ProtocolVersion::V334: {
      protocol::AdmaDataV334 packet{};
      std::memcpy(&packet, payload.data(), sizeof(packet));
      out.kinematics = fromV334(packet);
      out.v334 = packet;
      return out;
    }
    case ProtocolVersion::V335: {
      protocol::AdmaDataV335 packet{};
      std::memcpy(&packet, payload.data(), sizeof(packet));
      out.kinematics = fromV335(packet);
      out.v335 = packet;
      return out;
    }
    default:
      throw std::runtime_error("Unsupported protocol version");
  }
}

ProtocolVersion AdmaPacketDecoder::parseVersion(std::string_view value) {
  if (value == "v3.2") {
    return ProtocolVersion::V32;
  }
  if (value == "v3.3.3") {
    return ProtocolVersion::V333;
  }
  if (value == "v3.3.4") {
    return ProtocolVersion::V334;
  }
  if (value == "v3.3.5") {
    return ProtocolVersion::V335;
  }
  throw std::runtime_error("Unsupported protocol string: " + std::string(value));
}

std::string AdmaPacketDecoder::toString(ProtocolVersion version) {
  switch (version) {
    case ProtocolVersion::V32:
      return "v3.2";
    case ProtocolVersion::V333:
      return "v3.3.3";
    case ProtocolVersion::V334:
      return "v3.3.4";
    case ProtocolVersion::V335:
      return "v3.3.5";
    default:
      return "unknown";
  }
}

std::size_t AdmaPacketDecoder::expectedPacketSize(ProtocolVersion version) {
  switch (version) {
    case ProtocolVersion::V32:
      return sizeof(protocol::AdmaDataV32);
    case ProtocolVersion::V333:
      return sizeof(protocol::AdmaDataV333);
    case ProtocolVersion::V334:
      return sizeof(protocol::AdmaDataV334);
    case ProtocolVersion::V335:
      return sizeof(protocol::AdmaDataV335);
    default:
      throw std::runtime_error("Unsupported protocol version");
  }
}

double AdmaPacketDecoder::scale(int32_t raw, double factor) {
  return static_cast<double>(raw) * factor;
}

DecodedKinematics AdmaPacketDecoder::fromV32(const protocol::AdmaDataV32& packet) {
  const double accx = scale(packet.accBody.x, 0.0004);
  const double accy = scale(packet.accBody.y, 0.0004);
  const double accz = scale(packet.accBody.z, 0.0004);
  const double vx_ms = scale(packet.gpsvelframex, 0.005);
  const double vy_ms = scale(packet.gpsvelframey, 0.005);
  return {accx, accy, accz, speedKmhFromMs(vx_ms, vy_ms)};
}

DecodedKinematics AdmaPacketDecoder::fromV333(const protocol::AdmaDataV333& packet) {
  const double accx = scale(packet.accBody.x, 0.0004);
  const double accy = scale(packet.accBody.y, 0.0004);
  const double accz = scale(packet.accBody.z, 0.0004);
  const double vx_ms = scale(packet.gnssvelframex, 0.005);
  const double vy_ms = scale(packet.gnssvelframey, 0.005);
  return {accx, accy, accz, speedKmhFromMs(vx_ms, vy_ms)};
}

DecodedKinematics AdmaPacketDecoder::fromV334(const protocol::AdmaDataV334& packet) {
  const double accx = scale(packet.accBody.x, 0.0004);
  const double accy = scale(packet.accBody.y, 0.0004);
  const double accz = scale(packet.accBody.z, 0.0004);
  const double vx_ms = scale(packet.insVelFrame.x, 0.005);
  const double vy_ms = scale(packet.insVelFrame.y, 0.005);
  return {accx, accy, accz, speedKmhFromMs(vx_ms, vy_ms)};
}

DecodedKinematics AdmaPacketDecoder::fromV335(const protocol::AdmaDataV335& packet) {
  const double accx = scale(packet.accBody.x, 0.0004);
  const double accy = scale(packet.accBody.y, 0.0004);
  const double accz = scale(packet.accBody.z, 0.0004);
  const double vx_ms = scale(packet.insVelFrame.x, 0.005);
  const double vy_ms = scale(packet.insVelFrame.y, 0.005);
  return {accx, accy, accz, speedKmhFromMs(vx_ms, vy_ms)};
}

}  // namespace adma
