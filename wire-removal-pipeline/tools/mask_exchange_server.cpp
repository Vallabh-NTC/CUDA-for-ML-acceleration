#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cerrno>

static void die(const char* msg){ perror(msg); std::exit(1); }

static int listen_on(uint16_t port){
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if(s<0) die("socket");
    int yes=1; if (setsockopt(s,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes))<0) die("setsockopt");
    sockaddr_in addr{}; addr.sin_family=AF_INET; addr.sin_addr.s_addr=INADDR_ANY; addr.sin_port=htons(port);
    if(bind(s,(sockaddr*)&addr,sizeof(addr))<0) die("bind");
    if(listen(s,1)<0) die("listen");
    std::cout << "[server] listening on " << port << std::endl;
    sockaddr_in cli{}; socklen_t cl=sizeof(cli);
    int c = accept(s,(sockaddr*)&cli,&cl);
    if(c<0) die("accept");
    close(s);
    return c;
}
static void send_all(int fd, const void* buf, size_t n){
    const uint8_t* p=(const uint8_t*)buf; size_t sent=0;
    while(sent<n){ ssize_t k=send(fd,p+sent,n-sent,0); if(k<=0) die("send"); sent+=k; }
}
static void recv_all(int fd, void* buf, size_t n){
    uint8_t* p=(uint8_t*)buf; size_t got=0;
    while(got<n){ ssize_t k=recv(fd,p+got,n-got,0); if(k<=0) die("recv"); got+=k; }
}

static constexpr uint32_t fourcc_u32(const char (&tag)[5]) noexcept {
    return  (uint32_t)(unsigned char)tag[0]
          | ((uint32_t)(unsigned char)tag[1] << 8)
          | ((uint32_t)(unsigned char)tag[2] << 16)
          | ((uint32_t)(unsigned char)tag[3] << 24);
}

#pragma pack(push,1)
struct FrameHeader {
    uint32_t magic, width, height, stride, channels, size;
};
struct StampsCountHeader {
    uint32_t magic;
    uint32_t count;
};
struct Stamp {
    uint32_t magic;     // "STMP"
    float    cx, cy;
    float    angle_deg;
    float    sep_px;
    uint32_t bw, bh;
};
#pragma pack(pop)

static const uint32_t FRAM_MAGIC = fourcc_u32("FRAM");
static const uint32_t SCNT_MAGIC = fourcc_u32("SCNT");
static const uint32_t STMP_MAGIC = fourcc_u32("STMP");

// config key (angle, sep, bw, bh) for grouping
struct CfgKey {
    int angle_mdeg;   // angle in milli-deg for stable hashing
    int sep_mpx;      // sep*1000
    uint32_t bw, bh;
    bool operator==(const CfgKey& o) const {
        return angle_mdeg==o.angle_mdeg && sep_mpx==o.sep_mpx && bw==o.bw && bh==o.bh;
    }
};
struct CfgKeyHash {
    size_t operator()(const CfgKey& k) const {
        size_t h = std::hash<int>()(k.angle_mdeg);
        h = h*1315423911u ^ std::hash<int>()(k.sep_mpx);
        h = h*1315423911u ^ std::hash<uint32_t>()(k.bw);
        h = h*1315423911u ^ std::hash<uint32_t>()(k.bh);
        return h;
    }
};

static inline int clampi(int v,int a,int b){ return v<a?a:(v>b?b:v); }

int main(int argc, char** argv){
    if(argc<3){ std::cerr<<"usage: "<<argv[0]<<" <video.mp4> <port>\n"; return 1; }
    const char* mp4 = argv[1];
    uint16_t port = (uint16_t)std::stoi(argv[2]);

    gst_init(&argc,&argv);

    // Decode → RGBA → appsink
    std::string pipe =
        std::string("filesrc location=") + mp4 +
        " ! qtdemux ! h264parse ! nvv4l2decoder "
        "! nvvidconv ! video/x-raw,format=RGBA "
        "! appsink name=sink sync=false max-buffers=32 drop=false";
    GError* err=nullptr;
    GstElement* p = gst_parse_launch(pipe.c_str(), &err);
    if(!p || err){ std::cerr<<"pipeline error: "<<(err?err->message:"")<<"\n"; return 2; }

    GstElement* sink_el = gst_bin_get_by_name(GST_BIN(p),"sink");
    if(!sink_el){ std::cerr<<"appsink not found\n"; return 2; }
    gst_element_set_state(p, GST_STATE_PLAYING);

    // Pull a later frame
    GstAppSink* sink = GST_APP_SINK(sink_el);
    GstSample* samp = nullptr;
    const gint64 TIMEOUT = 5 * GST_SECOND;
    for(int i=0;i<60;i++){
        GstSample* next = gst_app_sink_try_pull_sample(sink, TIMEOUT);
        if(!next){ std::cerr<<"[server] timed out pulling sample\n"; return 3; }
        if(samp) gst_sample_unref(samp);
        samp = next;
    }

    GstCaps* caps = gst_sample_get_caps(samp);
    GstStructure* s = gst_caps_get_structure(caps,0);
    int w=0,h=0; gst_structure_get_int(s,"width",&w); gst_structure_get_int(s,"height",&h);

    GstBuffer* buf = gst_sample_get_buffer(samp);
    GstMapInfo mi;
    if(!gst_buffer_map(buf,&mi,GST_MAP_READ)){ std::cerr<<"gst_buffer_map failed\n"; return 3; }
    const uint8_t* rgba = mi.data;
    const uint32_t stride = (uint32_t)w * 4u;
    const uint32_t size   = (uint32_t)h * stride;

    // TCP to client
    int cfd = listen_on(port);

    // Send frame header + payload (LE)
    FrameHeader fh{};
    fh.magic=FRAM_MAGIC; fh.width=(uint32_t)w; fh.height=(uint32_t)h; fh.stride=stride; fh.channels=4; fh.size=size;
    send_all(cfd,&fh,sizeof(fh));
    send_all(cfd,rgba,size);

    // ---- receive stamps -------------------------------------------------
    StampsCountHeader scnt{};
    recv_all(cfd, &scnt, sizeof(scnt));
    if(scnt.magic != SCNT_MAGIC){
        std::cerr << "[server] protocol error: expected SCNT magic, got " << std::hex << scnt.magic << std::dec << "\n";
        close(cfd);
        return 4;
    }
    std::cout << "[server] expecting " << scnt.count << " stamp(s)\n";

    std::vector<Stamp> stamps;
    stamps.reserve(scnt.count);

    for(uint32_t i=0;i<scnt.count;i++){
        Stamp st{};
        recv_all(cfd, &st, sizeof(st));
        if(st.magic != STMP_MAGIC){
            std::cerr << "[server] bad STMP at index " << i << " (magic="<< std::hex << st.magic << std::dec << ")\n";
            close(cfd);
            return 5;
        }
        // Light sanity checks (bounds are clamped later anyway)
        if (st.bw==0 || st.bh==0){
            std::cerr << "[server] warning: zero-sized block at " << i << " -> forcing to 1\n";
            st.bw = st.bw? st.bw : 1;
            st.bh = st.bh? st.bh : 1;
        }
        stamps.push_back(st);
    }
    close(cfd);

    // ---- dump exact received stamps (for debugging/traceability) --------
    {
        std::ofstream csv("/dev/shm/wire_stamps.csv");
        csv << "index,cx,cy,angle_deg,sep_px,bw,bh\n";
        for (size_t i=0;i<stamps.size();++i){
            const auto& st = stamps[i];
            csv << i << "," << st.cx << "," << st.cy << ","
                << st.angle_deg << "," << st.sep_px << ","
                << st.bw << "," << st.bh << "\n";
        }
        std::cout << "[server] wrote /dev/shm/wire_stamps.csv ("<<stamps.size()<<" rows)\n";
    }

    // ---- group stamps by config & rasterize masks -----------------------
    using MaskBuf = std::vector<uint8_t>;
    std::unordered_map<CfgKey, MaskBuf, CfgKeyHash> groups;
    groups.reserve(stamps.size() ? stamps.size() : 1);

    for(const auto& st : stamps){
        CfgKey key{
            (int)std::llround((double)st.angle_deg * 1000.0),   // mdeg
            (int)std::llround((double)st.sep_px   * 1000.0),    // mpx
            st.bw, st.bh
        };
        auto& mask = groups[key];
        if(mask.empty()) mask.assign((size_t)w*(size_t)h, 0);

        int cx = (int)std::lround(st.cx);
        int cy = (int)std::lround(st.cy);
        int bw = (int)st.bw;
        int bh = (int)st.bh;

        // Destination rectangle centered at (cx,cy)
        int x0 = clampi(cx - bw/2, 0, w-1);
        int y0 = clampi(cy - bh/2, 0, h-1);
        int x1 = clampi(cx + bw/2 + 1, 1, w);
        int y1 = clampi(cy + bh/2 + 1, 1, h);

        if (x1<=x0 || y1<=y0) continue; // fully clipped

        for(int yy=y0; yy<y1; ++yy){
            uint8_t* row = mask.data() + (size_t)yy * (size_t)w;
            std::memset(row + x0, 255, (size_t)(x1 - x0));
        }
    }

    // ---- compute (dx,dy) per group: vector from dest center to TOP donor
    struct OutItem { CfgKey key; MaskBuf mask; float dx, dy; };
    std::vector<OutItem> outs; outs.reserve(groups.size());
    for (auto& kv : groups){
        const auto& key = kv.first;
        auto&       m   = kv.second;

        double angle_deg = key.angle_mdeg / 1000.0;
        double sep_px    = key.sep_mpx  / 1000.0;
        double ang = angle_deg * M_PI / 180.0;
        double vx = std::cos(ang), vy = std::sin(ang);
        double half = sep_px * 0.5;

        float dx = (float)(-vx * half);   // top dot offset: (cx - vx*half, cy - vy*half)
        float dy = (float)(-vy * half);

        outs.push_back(OutItem{key, std::move(m), dx, dy});
    }

    // ---- write meta + masks ---------------------------------------------
    {
        std::ofstream meta("/dev/shm/wire_mask.meta");
        meta << w << " " << h << " " << outs.size() << "\n";
        for (size_t i=0;i<outs.size();++i){
            meta << outs[i].dx << " " << outs[i].dy << "\n";
        }
        std::cout << "[server] wrote /dev/shm/wire_mask.meta (groups="<<outs.size()<<")\n";
    }
    for (size_t i=0;i<outs.size(); ++i){
        char path[128];
        std::snprintf(path, sizeof(path), "/dev/shm/wire_mask_%03zu.raw", i+1);
        std::ofstream raw(path, std::ios::binary);
        raw.write(reinterpret_cast<const char*>(outs[i].mask.data()), outs[i].mask.size());
        std::cout << "[server] wrote "<<path<<" ("<<w<<"x"<<h<<")\n";
    }

    // cleanup
    gst_buffer_unmap(buf,&mi);
    gst_sample_unref(samp);
    gst_element_set_state(p,GST_STATE_NULL);
    gst_object_unref(p);
    std::cout << "[server] done\n";
    return 0;
}
