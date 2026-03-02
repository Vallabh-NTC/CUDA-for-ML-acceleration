#pragma once
#include <cstdint>

class ESP21 {
public:
    ESP21() = default;

    void decode(const unsigned char* buff);

    struct Data {
        float vehicle_speed;     // esp_v_signal.phy
        bool valid;
        uint8_t esp_intervention;
    };

    const Data& data() const { return d; }

private:
    Data d{};
};
