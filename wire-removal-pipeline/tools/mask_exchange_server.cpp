#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>
#include "../common/protocol.hpp"

static void die(const char* msg){ perror(msg); std::exit(1); }

static int listen_on(uint16_t port){
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if(s<0) die("socket");
    int yes=1; setsockopt(s,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
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
    gst_element_set_state(p, GST_STATE_PLAYING);

    // Pull a later frame to ensure a good sample (adjust as desired)
    GstAppSink* sink = GST_APP_SINK(sink_el);
    GstSample* samp = nullptr;
    const gint64 TIMEOUT = 5 * GST_SECOND;
    for(int i=0;i<60;i++){
        GstSample* next = gst_app_sink_try_pull_sample(sink, TIMEOUT);
        if(!next){
            std::cerr<<"[server] timed out pulling frame "<<(i+1)<<"\n";
            return 3;
        }
        if(samp) gst_sample_unref(samp);
        samp = next;
    }

    GstCaps* caps = gst_sample_get_caps(samp);
    GstStructure* s = gst_caps_get_structure(caps,0);
    int w=0,h=0; gst_structure_get_int(s,"width",&w); gst_structure_get_int(s,"height",&h);

    GstBuffer* buf = gst_sample_get_buffer(samp);
    GstMapInfo mi;
    if(!gst_buffer_map(buf,&mi,GST_MAP_READ)){ std::cerr<<"map failed\n"; return 3; }
    const uint8_t* rgba = mi.data;
    const uint32_t stride = w*4;
    const uint32_t size   = h*stride;

    // TCP to client
    int cfd = listen_on(port);

    // Send frame header + payload
    FrameHeader fh{};
    fh.magic=FRAM_MAGIC; fh.width=w; fh.height=h; fh.stride=stride; fh.channels=4; fh.size=size;
    send_all(cfd,&fh,sizeof(fh));
    send_all(cfd,rgba,size);

    // ---- receive multi-masks -------------------------------------------------
    // Expect MCNT
    MasksCountHeader mcnt{};
    recv_all(cfd, &mcnt, sizeof(mcnt));
    if(mcnt.magic != MCNT_MAGIC){
        std::cerr << "[server] protocol error: expected MCNT magic\n";
        close(cfd);
        return 4;
    }
    if(mcnt.count == 0){
        std::cerr << "[server] warning: MCNT reported 0 masks\n";
    }
    std::cout << "[server] expecting " << mcnt.count << " masks\n";

    std::vector<std::vector<uint8_t>> masks;
    masks.reserve(mcnt.count);

    for(uint32_t i=0;i<mcnt.count;i++){
        MaskHeader mh{};
        recv_all(cfd,&mh,sizeof(mh));
        if(mh.magic!=MASK_MAGIC || mh.width!=(uint32_t)w || mh.height!=(uint32_t)h || mh.size!=(uint32_t)(w*h)){
            std::cerr<<"[server] mask header mismatch at index "<<i<<"\n";
            close(cfd);
            return 5;
        }
        std::vector<uint8_t> m(mh.size);
        recv_all(cfd, m.data(), m.size());
        masks.emplace_back(std::move(m));
        std::cout << "[server] received mask " << (i+1) << "/" << mcnt.count << "\n";
    }
    close(cfd);

    // ---- Save masks for downstream plugin -----------------------------------
    // Write meta: width height count
    {
        std::ofstream meta("/dev/shm/wire_mask.meta");
        meta << w << " " << h << " " << masks.size() << "\n";
    }
    // Each mask as /dev/shm/wire_mask_###.raw
    for(size_t i=0;i<masks.size();++i){
        char path[128];
        std::snprintf(path, sizeof(path), "/dev/shm/wire_mask_%03zu.raw", i+1);
        std::ofstream raw(path, std::ios::binary);
        raw.write(reinterpret_cast<const char*>(masks[i].data()), masks[i].size());
    }
    std::cout<<"[server] saved "<<masks.size()<<" masks to /dev/shm/wire_mask_###.raw ("<<w<<"x"<<h<<")\n";

    gst_buffer_unmap(buf,&mi);
    gst_sample_unref(samp);
    gst_element_set_state(p,GST_STATE_NULL);
    gst_object_unref(p);
    return 0;
}
