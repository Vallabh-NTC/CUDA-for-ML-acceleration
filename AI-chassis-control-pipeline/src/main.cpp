// Signals: SARA_10, SARA_08, LWI01, Motor20, BremseEV01, ESP21

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <cmath>

#include "UdpReceiver.hpp"
#include "SARA_10.hpp"
#include "SARA_08.hpp"
#include "LWI01.hpp"
#include "Motor20.hpp"
#include "BrakeEV01.hpp"
#include "ESP21.hpp"

static inline void print_line(
    int cluster,
    float ax, float ay, float az,
    float ox, float oy, float oz,
    float steer, float steer_spd,
    float gas, float brake, float v,
    const timespec& ts)
{
    std::tm tm_local{};
    localtime_r(&ts.tv_sec, &tm_local);

    char tb[32];
    std::strftime(tb, sizeof(tb), "%H:%M:%S", &tm_local);
    long ms = ts.tv_nsec / 1000000;

    std::cout << std::fixed << std::setprecision(3)
              << "[" << tb << "." << std::setw(3) << std::setfill('0') << ms
              << std::setfill(' ') << "] "
              << "C" << cluster
              << " | ax=" << ax << " ay=" << ay << " az=" << az
              << " | ox=" << ox << " oy=" << oy << " oz=" << oz
              << " | steer=" << steer << " spd=" << steer_spd
              << " | gas=" << gas << " brake=" << brake
              << " v=" << v
              << "\n";
}

int main()
{
    UdpReceiver receiver(1500);
    unsigned char buf[65536];

    while (true) {
        int n = receiver.receive(buf, sizeof(buf));
        if (n < 3) continue;

        int cluster = buf[0];
        const unsigned char* pdus = buf + 3;

        timespec ts{};
        clock_gettime(CLOCK_REALTIME, &ts);

        float ax=NAN, ay=NAN, az=NAN;
        float ox=NAN, oy=NAN, oz=NAN;
        float steer=NAN, steer_spd=NAN;
        float gas=NAN, brake=NAN, v=NAN;

        bool has = false;

        switch (cluster) {

        // ---------------- Cluster 1 ----------------
        case 1: {
            BrakeEV01 br; br.decode(pdus + 364); brake = br.brake_percent;
            SARA_10 s10; s10.decode(pdus + 412);
            SARA_08 s08; s08.decode(pdus + 437);
            ESP21 e; e.decode(pdus + 542); v = e.data().vehicle_speed;
            Motor20 m; m.decode(pdus + 629); gas = m.data().gas_percent;
            LWI01 l; l.decode(pdus + 677); steer = l.angle; steer_spd = l.speed;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 2 ----------------
        case 2: {
            SARA_10 s10; s10.decode(pdus + 274);
            SARA_08 s08; s08.decode(pdus + 348);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 3 ----------------
        case 3: {
            LWI01 l; l.decode(pdus + 56); steer = l.angle; steer_spd = l.speed;
            BrakeEV01 br; br.decode(pdus + 261); brake = br.brake_percent;
            SARA_10 s10; s10.decode(pdus + 326);
            SARA_08 s08; s08.decode(pdus + 425);
            Motor20 m; m.decode(pdus + 434); gas = m.data().gas_percent;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 4 ----------------
        case 4: {
            SARA_10 s10; s10.decode(pdus + 128);
            SARA_08 s08; s08.decode(pdus + 385);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 5 ----------------
        case 5: {
            BrakeEV01 br; br.decode(pdus + 48); brake = br.brake_percent;
            SARA_10 s10; s10.decode(pdus + 213);
            SARA_08 s08; s08.decode(pdus + 230);
            ESP21 e; e.decode(pdus + 375); v = e.data().vehicle_speed;
            LWI01 l; l.decode(pdus + 470); steer = l.angle; steer_spd = l.speed;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 6 ----------------
        case 6: {
            SARA_10 s10; s10.decode(pdus + 48);
            SARA_08 s08; s08.decode(pdus + 73);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 7 ----------------
        case 7: {
            LWI01 l; l.decode(pdus + 96); steer = l.angle; steer_spd = l.speed;
            BrakeEV01 br; br.decode(pdus + 196); brake = br.brake_percent;
            Motor20 m; m.decode(pdus + 233); gas = m.data().gas_percent;
            SARA_08 s08; s08.decode(pdus + 241);
            SARA_10 s10; s10.decode(pdus + 366);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 8 ----------------
        case 8: {
            SARA_08 s08; s08.decode(pdus + 349);
            SARA_10 s10; s10.decode(pdus + 1202);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 9 ----------------
        case 9: {
            ESP21 e; e.decode(pdus + 0); v = e.data().vehicle_speed;
            Motor20 m; m.decode(pdus + 204); gas = m.data().gas_percent;
            LWI01 l; l.decode(pdus + 172); steer = l.angle; steer_spd = l.speed;
            BrakeEV01 br; br.decode(pdus + 536); brake = br.brake_percent;
            SARA_08 s08; s08.decode(pdus + 560);
            SARA_10 s10; s10.decode(pdus + 620);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 10 ----------------
        case 10: {
            SARA_10 s10; s10.decode(pdus + 420);
            SARA_08 s08; s08.decode(pdus + 849);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 11 ----------------
        case 11: {
            SARA_10 s10; s10.decode(pdus + 27);
            BrakeEV01 br; br.decode(pdus + 537); brake = br.brake_percent;
            SARA_08 s08; s08.decode(pdus + 561);
            LWI01 l; l.decode(pdus + 1142); steer = l.angle; steer_spd = l.speed;
            Motor20 m; m.decode(pdus + 1158); gas = m.data().gas_percent;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 12 ----------------
        case 12: {
            SARA_08 s08; s08.decode(pdus + 403);
            SARA_10 s10; s10.decode(pdus + 436);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 13 ----------------
        case 13: {
            Motor20 m; m.decode(pdus + 79); gas = m.data().gas_percent;
            ESP21 e; e.decode(pdus + 150); v = e.data().vehicle_speed;
            LWI01 l; l.decode(pdus + 348); steer = l.angle; steer_spd = l.speed;
            SARA_10 s10; s10.decode(pdus + 671);
            SARA_08 s08; s08.decode(pdus + 820);
            BrakeEV01 br; br.decode(pdus + 958); brake = br.brake_percent;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 14 ----------------
        case 14: {
            SARA_08 s08; s08.decode(pdus + 121);
            az = s08.data().accel_z;
            ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 15 ----------------
        case 15: {
            SARA_10 s10; s10.decode(pdus + 130);
            SARA_08 s08; s08.decode(pdus + 192);
            BrakeEV01 br; br.decode(pdus + 254); brake = br.brake_percent;
            Motor20 m; m.decode(pdus + 612); gas = m.data().gas_percent;
            LWI01 l; l.decode(pdus + 715); steer = l.angle; steer_spd = l.speed;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 16 ----------------
        case 16: {
            SARA_08 s08; s08.decode(pdus + 866);
            SARA_10 s10; s10.decode(pdus + 1187);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 17 ----------------
        case 17: {
            SARA_08 s08; s08.decode(pdus + 125);
            SARA_10 s10; s10.decode(pdus + 340);
            Motor20 m; m.decode(pdus + 349); gas = m.data().gas_percent;
            BrakeEV01 br; br.decode(pdus + 593); brake = br.brake_percent;
            LWI01 l; l.decode(pdus + 637); steer = l.angle; steer_spd = l.speed;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 18 ----------------
        case 18: {
            SARA_10 s10; s10.decode(pdus + 749);
            SARA_08 s08; s08.decode(pdus + 1114);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 19 ----------------
        case 19: {
            BrakeEV01 br; br.decode(pdus + 177); brake = br.brake_percent;
            Motor20 m; m.decode(pdus + 359); gas = m.data().gas_percent;
            LWI01 l; l.decode(pdus + 416); steer = l.angle; steer_spd = l.speed;
            SARA_10 s10; s10.decode(pdus + 522);
            SARA_08 s08; s08.decode(pdus + 813);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 20 ----------------
        case 20: {
            SARA_10 s10; s10.decode(pdus + 214);
            SARA_08 s08; s08.decode(pdus + 260);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 21 ----------------
        case 21: {
            SARA_08 s08; s08.decode(pdus + 45);
            SARA_10 s10; s10.decode(pdus + 792);
            Motor20 m; m.decode(pdus + 689); gas = m.data().gas_percent;
            BrakeEV01 br; br.decode(pdus + 1235); brake = br.brake_percent;
            LWI01 l; l.decode(pdus + 242); steer = l.angle; steer_spd = l.speed;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 22 ----------------
        case 22: {
            SARA_10 s10; s10.decode(pdus + 207);
            SARA_08 s08; s08.decode(pdus + 374);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 23 ----------------
        case 23: {
            SARA_08 s08; s08.decode(pdus + 206);
            BrakeEV01 br; br.decode(pdus + 284); brake = br.brake_percent;
            Motor20 m; m.decode(pdus + 598); gas = m.data().gas_percent;
            SARA_10 s10; s10.decode(pdus + 552);
            LWI01 l; l.decode(pdus + 802); steer = l.angle; steer_spd = l.speed;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 24 ----------------
        case 24: {
            SARA_08 s08; s08.decode(pdus + 77);
            SARA_10 s10; s10.decode(pdus + 856);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 25 ----------------
        case 25: {
            SARA_10 s10; s10.decode(pdus + 48);
            SARA_08 s08; s08.decode(pdus + 86);
            Motor20 m; m.decode(pdus + 78); gas = m.data().gas_percent;
            BrakeEV01 br; br.decode(pdus + 661); brake = br.brake_percent;
            LWI01 l; l.decode(pdus + 162); steer = l.angle; steer_spd = l.speed;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 26 ----------------
        case 26: {
            SARA_08 s08; s08.decode(pdus + 273);
            SARA_10 s10; s10.decode(pdus + 894);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 27 ----------------
        case 27: {
            BrakeEV01 br; br.decode(pdus + 392); brake = br.brake_percent;
            Motor20 m; m.decode(pdus + 617); gas = m.data().gas_percent;
            LWI01 l; l.decode(pdus + 625); steer = l.angle; steer_spd = l.speed;
            SARA_10 s10; s10.decode(pdus + 682);
            SARA_08 s08; s08.decode(pdus + 1130);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 28 ----------------
        case 28: {
            SARA_10 s10; s10.decode(pdus + 8);
            SARA_08 s08; s08.decode(pdus + 70);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 29 ----------------
        case 29: {
            SARA_08 s08; s08.decode(pdus + 206);
            BrakeEV01 br; br.decode(pdus + 422); brake = br.brake_percent;
            SARA_10 s10; s10.decode(pdus + 596);
            Motor20 m; m.decode(pdus + 1003); gas = m.data().gas_percent;
            LWI01 l; l.decode(pdus + 1027); steer = l.angle; steer_spd = l.speed;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 30 ----------------
        case 30: {
            SARA_08 s08; s08.decode(pdus + 288);
            SARA_10 s10; s10.decode(pdus + 457);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 31 ----------------
        case 31: {
            BrakeEV01 br; br.decode(pdus + 16); brake = br.brake_percent;
            LWI01 l; l.decode(pdus + 32); steer = l.angle; steer_spd = l.speed;
            Motor20 m; m.decode(pdus + 179); gas = m.data().gas_percent;
            SARA_10 s10; s10.decode(pdus + 367);
            SARA_08 s08; s08.decode(pdus + 125);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 32 ----------------
        case 32: {
            SARA_08 s08; s08.decode(pdus + 654);
            SARA_10 s10; s10.decode(pdus + 684);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 33 ----------------
        case 33: {
            BrakeEV01 br; br.decode(pdus + 172); brake = br.brake_percent;
            Motor20 m; m.decode(pdus + 559); gas = m.data().gas_percent;
            LWI01 l; l.decode(pdus + 575); steer = l.angle; steer_spd = l.speed;
            SARA_08 s08; s08.decode(pdus + 518);
            SARA_10 s10; s10.decode(pdus + 631);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 34 ----------------
        case 34: {
            SARA_10 s10; s10.decode(pdus + 348);
            SARA_08 s08; s08.decode(pdus + 661);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 35 ----------------
        case 35: {
            BrakeEV01 br; br.decode(pdus + 116); brake = br.brake_percent;
            Motor20 m; m.decode(pdus + 65); gas = m.data().gas_percent;
            SARA_10 s10; s10.decode(pdus + 457);
            SARA_08 s08; s08.decode(pdus + 614);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 36 ----------------
        case 36: {
            SARA_08 s08; s08.decode(pdus + 32);
            SARA_10 s10; s10.decode(pdus + 306);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 37 ----------------
        case 37: {
            SARA_10 s10; s10.decode(pdus + 222);
            SARA_08 s08; s08.decode(pdus + 252);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 38 ----------------
        case 38: {
            SARA_10 s10; s10.decode(pdus + 111);
            SARA_08 s08; s08.decode(pdus + 759);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 39 ----------------
        case 39: {
            SARA_10 s10; s10.decode(pdus + 373);
            SARA_08 s08; s08.decode(pdus + 380);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 40 ----------------
        case 40: {
            SARA_10 s10; s10.decode(pdus + 385);
            SARA_08 s08; s08.decode(pdus + 402);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 41 ----------------
        case 41: {
            SARA_10 s10; s10.decode(pdus + 692);
            SARA_08 s08; s08.decode(pdus + 747);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 42 ----------------
        case 42: {
            SARA_10 s10; s10.decode(pdus + 518);
            SARA_08 s08; s08.decode(pdus + 520);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 43 ----------------
        case 43: {
            SARA_10 s10; s10.decode(pdus + 522);
            SARA_08 s08; s08.decode(pdus + 528);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 44 ----------------
        case 44: {
            SARA_10 s10; s10.decode(pdus + 589);
            SARA_08 s08; s08.decode(pdus + 595);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 45 ----------------
        case 45: {
            SARA_10 s10; s10.decode(pdus + 609);
            SARA_08 s08; s08.decode(pdus + 617);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 46 ----------------
        case 46: {
            SARA_10 s10; s10.decode(pdus + 620);
            SARA_08 s08; s08.decode(pdus + 652);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 47 ----------------
        case 47: {
            SARA_10 s10; s10.decode(pdus + 671);
            SARA_08 s08; s08.decode(pdus + 682);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 48 ----------------
        case 48: {
            SARA_10 s10; s10.decode(pdus + 684);
            SARA_08 s08; s08.decode(pdus + 692);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 49 ----------------
        case 49: {
            SARA_10 s10; s10.decode(pdus + 747);
            SARA_08 s08; s08.decode(pdus + 792);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 50 ----------------
        case 50: {
            SARA_10 s10; s10.decode(pdus + 792);
            SARA_08 s08; s08.decode(pdus + 856);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 51 ----------------
        case 51: {
            SARA_10 s10; s10.decode(pdus + 856);
            SARA_08 s08; s08.decode(pdus + 1187);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 52 ----------------
        case 52: {
            SARA_10 s10; s10.decode(pdus + 1187);
            SARA_08 s08; s08.decode(pdus + 1202);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 53 ----------------
        case 53: {
            SARA_10 s10; s10.decode(pdus + 0);
            SARA_08 s08; s08.decode(pdus + 16);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 54 ----------------
        case 54: {
            SARA_10 s10; s10.decode(pdus + 16);
            SARA_08 s08; s08.decode(pdus + 27);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 55 ----------------
        case 55: {
            SARA_10 s10; s10.decode(pdus + 27);
            SARA_08 s08; s08.decode(pdus + 48);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 56 ----------------
        case 56: {
            SARA_10 s10; s10.decode(pdus + 48);
            SARA_08 s08; s08.decode(pdus + 64);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 57 ----------------
        case 57: {
            SARA_10 s10; s10.decode(pdus + 64);
            SARA_08 s08; s08.decode(pdus + 80);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 58 ----------------
        case 58: {
            SARA_10 s10; s10.decode(pdus + 80);
            SARA_08 s08; s08.decode(pdus + 95);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 59 ----------------
        case 59: {
            SARA_10 s10; s10.decode(pdus + 95);
            SARA_08 s08; s08.decode(pdus + 111);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 60 ----------------
        case 60: {
            SARA_10 s10; s10.decode(pdus + 112);
            SARA_08 s08; s08.decode(pdus + 118);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 61 ----------------
        case 61: {
            SARA_10 s10; s10.decode(pdus + 118);
            SARA_08 s08; s08.decode(pdus + 128);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 62 ----------------
        case 62: {
            SARA_10 s10; s10.decode(pdus + 128);
            SARA_08 s08; s08.decode(pdus + 130);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 63 ----------------
        case 63: {
            SARA_10 s10; s10.decode(pdus + 16);
            BrakeEV01 br; br.decode(pdus + 57); brake = br.brake_percent;
            SARA_08 s08; s08.decode(pdus + 161);
            Motor20 m; m.decode(pdus + 170); gas = m.data().gas_percent;
            LWI01 l; l.decode(pdus + 250); steer = l.angle; steer_spd = l.speed;

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        // ---------------- Cluster 64 ----------------
        case 64: {
            SARA_10 s10; s10.decode(pdus + 341);
            SARA_08 s08; s08.decode(pdus + 468);

            ax = s10.data().accel_x; ay = s10.data().accel_y; oz = s10.data().omega_z;
            az = s08.data().accel_z; ox = s08.data().omega_x; oy = s08.data().omega_y;
            has = true;
            break;
        }

        default:
            break;
        }

        if (has) {
            print_line(cluster,
                       ax, ay, az,
                       ox, oy, oz,
                       steer, steer_spd,
                       gas, brake, v,
                       ts);
        }
    }
}

