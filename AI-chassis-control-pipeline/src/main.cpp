#include "UdpReceiver.hpp"
#include "LWI01.hpp"
#include "Lichthinten01.hpp"
#include <iostream>

int main() {
    UdpReceiver receiver(1500);
    unsigned char buf[4096];

    std::cout << "AAAAAAAAA" << std::endl;  // oppure: << std::flush;

    while(true) {
        int n = receiver.receive(buf, sizeof(buf));
        if(n <= 0) continue;

        if(n < 700) continue;

        int cluster = buf[0];
        if(cluster != 1) continue;

        const unsigned char* pdus = buf + 3;

        LWI01 lwi;
        lwi.decode(pdus + 677);
        lwi.print();

        Lichthinten01 lh;
        lh.decode(pdus + 605);
        lh.print();
    }
}
