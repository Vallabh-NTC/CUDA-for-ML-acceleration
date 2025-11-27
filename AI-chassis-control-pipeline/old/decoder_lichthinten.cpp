#include <iostream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <cstdint>

#pragma comment(lib, "Ws2_32.lib")

// ----------------------- Bit Extractor -------------------------
static uint64_t extract_bits(const unsigned char *buf, int byte, int bit, int len)
{
    uint64_t val = 0;
    int bitpos = byte * 8 + bit;

    for(int i = 0; i < len; i++)
    {
        int b = bitpos + i;
        int srcByte = b / 8;
        int srcBit  = b % 8;
        uint8_t bitval = (buf[srcByte] >> srcBit) & 1;
        val |= (uint64_t)bitval << i;
    }
    return val;
}

int main()
{
    WSADATA wsa;
    WSAStartup(MAKEWORD(2,2), &wsa);

    SOCKET sock = socket(AF_INET, SOCK_DGRAM, 0);

    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(1500);

    bind(sock, (sockaddr *)&addr, sizeof(addr));
    std::cout << "Listening for UDP packets on port 1500...\n";

    unsigned char buf[4096];

    while(true)
    {
        int n = recv(sock, (char*)buf, sizeof(buf), 0);
        if(n <= 0) continue;

        if(n < 700)
            continue;

        int cluster = buf[0];
        if(cluster != 1)
            continue;

        const unsigned char* pdus = buf + 3;

        // YOUR OFFSET
        const unsigned char* lh = pdus + 605;

        // ----------- Decode all Lichthinten01 signals -------------
        uint8_t bcm2_bremsl_durch_ecd = extract_bits(lh, 0, 5, 1);
        uint8_t lh_aussenlicht_def     = extract_bits(lh, 0, 7, 1);

        uint8_t lh_standlicht_h_aktiv  = extract_bits(lh, 1, 0, 1);
        uint8_t lh_parklicht_hl_aktiv  = extract_bits(lh, 1, 1, 1);
        uint8_t lh_parklicht_hr_aktiv  = extract_bits(lh, 1, 2, 1);
        uint8_t lh_bremslicht_h_aktiv  = extract_bits(lh, 1, 3, 1);
        uint8_t lh_nebelschluss_aktiv  = extract_bits(lh, 1, 4, 1);
        uint8_t lh_rueckfahrlicht_aktiv= extract_bits(lh, 1, 5, 1);
        uint8_t lh_blinker_hl_akt      = extract_bits(lh, 1, 6, 1);
        uint8_t lh_blinker_hr_akt      = extract_bits(lh, 1, 7, 1);

        uint8_t lh_blinker_li_def      = extract_bits(lh, 2, 0, 1);
        uint8_t lh_bremsl_li_def       = extract_bits(lh, 2, 1, 1);
        uint8_t lh_schlusslicht_li_def = extract_bits(lh, 2, 2, 1);
        uint8_t lh_rueckf_li_def       = extract_bits(lh, 2, 3, 1);
        uint8_t lh_nebel_li_def        = extract_bits(lh, 2, 4, 1);
        uint8_t lh_schluss_brems_nebel_li_def = extract_bits(lh, 2, 5, 1);
        uint8_t lh_schluss_brems_nebel_re_def = extract_bits(lh, 2, 6, 1);
        uint8_t lh_zusatzschlussl_def  = extract_bits(lh, 2, 7, 1);

        uint8_t lh_schluss_brems_li_def = extract_bits(lh, 3, 0, 1);
        uint8_t lh_schluss_nebel_li_def = extract_bits(lh, 3, 1, 1);
        uint8_t lh_sl_brl_blk_li_def    = extract_bits(lh, 3, 2, 1);
        uint8_t lh_brems_blk_li_def     = extract_bits(lh, 3, 3, 1);
        uint8_t lh_diag_status_re_def   = extract_bits(lh, 3, 4, 1);
        uint8_t lh_diag_status_li_def   = extract_bits(lh, 3, 5, 1);
        uint8_t lh_diag_led_li_def      = extract_bits(lh, 3, 6, 1);
        uint8_t lh_diag_led_re_def      = extract_bits(lh, 3, 7, 1);

        uint8_t lh_blinker_re_def         = extract_bits(lh, 4, 0, 1);
        uint8_t lh_bremsl_re_def          = extract_bits(lh, 4, 1, 1);
        uint8_t lh_schlusslicht_re_def    = extract_bits(lh, 4, 2, 1);
        uint8_t lh_rueckf_re_def          = extract_bits(lh, 4, 3, 1);
        uint8_t lh_nebel_re_def           = extract_bits(lh, 4, 4, 1);
        uint8_t lh_schluss_brems_mi_def   = extract_bits(lh, 4, 5, 1);

        uint8_t lh_schluss_brems_re_def     = extract_bits(lh, 5, 0, 1);
        uint8_t lh_schluss_nebel_re_def     = extract_bits(lh, 5, 1, 1);
        uint8_t lh_sl_brl_blk_re_def        = extract_bits(lh, 5, 2, 1);
        uint8_t lh_brems_blk_re_def         = extract_bits(lh, 5, 3, 1);

        uint8_t lh_kennzl_def     = extract_bits(lh, 6, 0, 1);
        uint8_t lh_3_bremsl_def   = extract_bits(lh, 6, 1, 1);
        uint8_t lh_nebel_mi_def   = extract_bits(lh, 6, 2, 1);
        uint8_t lh_rueckf_mi_def  = extract_bits(lh, 6, 3, 1);
        uint8_t lh_schlusslicht_mi_def = extract_bits(lh, 6, 4, 1);
        uint8_t lh_bremsl_mi_def  = extract_bits(lh, 6, 5, 1);
        uint8_t lh_bremsl_li_ges_def = extract_bits(lh, 6, 6, 1);
        uint8_t lh_bremsl_re_ges_def = extract_bits(lh, 6, 7, 1);

        // -------- PRINT ------------
        std::cout << "\n--- LICHTHINTEN01 ---\n";
        std::cout << "Bremslicht hinten aktiv: " << int(lh_bremslicht_h_aktiv) << "\n";
        std::cout << "Rueckfahrlicht aktiv:    " << int(lh_rueckfahrlicht_aktiv) << "\n";
        std::cout << "Blinker HL:              " << int(lh_blinker_hl_akt) << "\n";
        std::cout << "Blinker HR:              " << int(lh_blinker_hr_akt) << "\n";
        std::cout << "Nebel hinten aktiv:      " << int(lh_nebelschluss_aktiv) << "\n";
        std::cout << "Parklicht HL:            " << int(lh_parklicht_hl_aktiv) << "\n";
        std::cout << "Parklicht HR:            " << int(lh_parklicht_hr_aktiv) << "\n";
        std::cout << "Kennzeichenlicht def:    " << int(lh_kennzl_def) << "\n";
        std::cout << "3. Bremslicht def:       " << int(lh_3_bremsl_def) << "\n";
    }

    closesocket(sock);
    WSACleanup();
    return 0;
}
