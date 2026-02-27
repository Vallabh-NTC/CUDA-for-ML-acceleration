#pragma once

#include <cstdint>

namespace adma::protocol {

#pragma pack(push, 1)

struct Reserved16 {
  char reservedSpace[16];
};

struct Reserved4 {
  char reservedSpace[4];
};

struct Reserved2 {
  char reservedSpace[2];
};

struct AdmaStaticHeader {
  char genesysid[4];
  char headerversion[4];
  Reserved16 reserved;
  uint32_t formatid;
  char formatversion[4];
  uint32_t serialno;
  char alias[32];
};

struct AdmaDynamicHeader {
  uint32_t configid;
  uint32_t configformat;
  uint32_t configversion;
  uint32_t configsize;
  uint32_t byteoffset;
  uint32_t slicesize;
  int32_t slicedata;
};

struct SensorBody {
  int32_t accHR;
  int32_t rateHR;
};

struct Vector3 {
  int16_t x;
  int16_t y;
  int16_t z;
  char reservedSpace[2];
};

struct Vector2 {
  int16_t x;
  int16_t y;
  char reservedSpace1[2];
  char reservedSpace2[2];
};

struct Miscellaneous {
  int16_t invPathRadius;
  int16_t sideSlipAngle;
  uint32_t distanceTraveled;
};

struct GNSSPosition {
  int32_t latitude;
  int32_t longitude;
};

struct INSPosition {
  GNSSPosition pos_abs;
  int32_t pos_rel_x;
  int32_t pos_rel_y;
};

#pragma pack(pop)

}  // namespace adma::protocol
