#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <nvbufsurface.h>

#include <vpi/VPI.h>
#include <vpi/Image.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/OpticalFlowDense.h>

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

#define CHECK_VPI(stmt)                                                         \
    do {                                                                        \
        VPIStatus _st = (stmt);                                                 \
        if (_st != VPI_SUCCESS) {                                               \
            std::cerr << "VPI error: " << vpiStatusGetName(_st)                 \
                      << " at " << #stmt << "\n";                               \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

static void print_caps_once(GstSample* sample)
{
    GstCaps* caps = gst_sample_get_caps(sample);
    if (!caps) return;
    gchar* s = gst_caps_to_string(caps);
    std::cout << "Caps: " << s << "\n";
    g_free(s);
}

static NvBufSurface* map_to_nvbufsurface(GstBuffer* buf, GstMapInfo& map)
{
    std::memset(&map, 0, sizeof(map));
    if (!gst_buffer_map(buf, &map, GST_MAP_READ))
        return nullptr;
    return reinterpret_cast<NvBufSurface*>(map.data);
}

static inline float s10_5_to_px(int16_t v) { return float(v) / 32.0f; }

// ---- Drawing helpers (BGR) ----
static void put_px(uint8_t *bgr, int W, int H, int stride, int x, int y,
                   uint8_t b, uint8_t g, uint8_t r)
{
    if (x < 0 || x >= W || y < 0 || y >= H) return;
    uint8_t *p = bgr + y*stride + x*3;
    p[0]=b; p[1]=g; p[2]=r;
}

// Bresenham line with thickness (square brush)
static void draw_line_bgr(uint8_t *bgr, int W, int H, int stride,
                          int x0, int y0, int x1, int y1,
                          uint8_t b, uint8_t g, uint8_t r,
                          int thickness)
{
    if (thickness < 1) thickness = 1;
    int half = thickness / 2;

    int dx = std::abs(x1-x0), sx = x0<x1 ? 1 : -1;
    int dy = -std::abs(y1-y0), sy = y0<y1 ? 1 : -1;
    int err = dx + dy;

    while (true) {
        for (int oy = -half; oy <= half; ++oy)
            for (int ox = -half; ox <= half; ++ox)
                put_px(bgr, W, H, stride, x0+ox, y0+oy, b,g,r);

        if (x0==x1 && y0==y1) break;
        int e2 = 2*err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
        if (x0<0||x0>=W||y0<0||y0>=H) break;
    }
}

static void draw_arrow_bgr(uint8_t *bgr, int W, int H, int stride,
                           int x0, int y0, int x1, int y1,
                           uint8_t b, uint8_t g, uint8_t r,
                           int thickness)
{
    draw_line_bgr(bgr, W, H, stride, x0,y0,x1,y1, b,g,r, thickness);

    // arrow head
    float ang = std::atan2(float(y1-y0), float(x1-x0));
    float len = 10.0f + 2.0f*thickness;

    int hx1 = int(x1 - len*std::cos(ang - 0.5f));
    int hy1 = int(y1 - len*std::sin(ang - 0.5f));
    int hx2 = int(x1 - len*std::cos(ang + 0.5f));
    int hy2 = int(y1 - len*std::sin(ang + 0.5f));

    draw_line_bgr(bgr, W, H, stride, x1,y1,hx1,hy1, b,g,r, thickness);
    draw_line_bgr(bgr, W, H, stride, x1,y1,hx2,hy2, b,g,r, thickness);
}

int main(int argc, char** argv)
{
    gst_init(&argc, &argv);

    // ---- Network destination ----
    const char* DEST_IP   = "192.168.1.100";
    const int   DEST_PORT = 5000;

    // ---- Video ----
    const int W = 2560;
    const int H = 720;
    const int FPS_NUM = 30;
    const int FPS_DEN = 1;

    // ---- OF parameters ----
    const int grid = 4;          // motion cell = 4x4 pixels
    const int numLevels = 1;
    int gridArr[1] = { grid };

    // ---- Drawing parameters ----
    // Fine field: thin green arrows directly from mvImg (4x4 cell grid)
    const int   fineStrideCells = 4;   // draw every 4 motion cells (=> every 16 pixels)
    const float fineScale = 8.0f;
    const float fineMinMag = 0.2f;     // px/frame
    const int   fineThickness = 1;

    // Coarse field: 16x16 blocks from averaging 4x4 motion cells
    const int   groupCells = 4;        // 4 cells * grid(4px) = 16 pixels
    const int   coarseStrideGroups = 2; // draw every 2 groups (=> every 32 pixels); set 1 for denser
    const float coarseScale = 10.0f;
    const float coarseMinMag = 0.2f;   // px/frame
    const int   coarseThickness = 3;   // thicker red arrows

    // -------------------------
    // Input pipeline (USB -> NVMM NV12 -> appsink)
    // -------------------------
    const char* PIPE_IN =
        "v4l2src device=/dev/video0 io-mode=2 ! "
        "video/x-raw,format=YUY2,width=2560,height=720,framerate=30/1 ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width=2560,height=720,framerate=30/1 ! "
        "appsink name=flowsink sync=false max-buffers=1 drop=true";

    GError* err = nullptr;
    GstElement* inpipe = gst_parse_launch(PIPE_IN, &err);
    if (!inpipe) {
        std::cerr << "gst_parse_launch(in) failed: " << (err ? err->message : "") << "\n";
        if (err) g_error_free(err);
        return 1;
    }
    GstElement* appsink = gst_bin_get_by_name(GST_BIN(inpipe), "flowsink");
    if (!appsink) {
        std::cerr << "appsink not found\n";
        gst_object_unref(inpipe);
        return 1;
    }
    gst_app_sink_set_drop(GST_APP_SINK(appsink), TRUE);
    gst_app_sink_set_max_buffers(GST_APP_SINK(appsink), 1);

    // -------------------------
    // Output pipeline (appsrc BGR -> H264 RTP -> UDP)
    // -------------------------
    std::string PIPE_OUT =
        std::string("appsrc name=mysrc is-live=true format=time do-timestamp=true ! ")
        + "video/x-raw,format=BGR,width=2560,height=720,framerate=30/1 ! "
        + "videoconvert ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
        + "nvv4l2h264enc bitrate=4000000 iframeinterval=15 idrinterval=15 insert-sps-pps=true maxperf-enable=true ! "
        + "h264parse config-interval=-1 ! rtph264pay pt=96 config-interval=1 ! "
        + "udpsink host=" + DEST_IP + " port=" + std::to_string(DEST_PORT) + " sync=false async=false";

    GstElement* outpipe = gst_parse_launch(PIPE_OUT.c_str(), &err);
    if (!outpipe) {
        std::cerr << "gst_parse_launch(out) failed: " << (err ? err->message : "") << "\n";
        if (err) g_error_free(err);
        gst_object_unref(appsink);
        gst_object_unref(inpipe);
        return 1;
    }
    GstElement* appsrc = gst_bin_get_by_name(GST_BIN(outpipe), "mysrc");
    if (!appsrc) {
        std::cerr << "appsrc not found\n";
        gst_object_unref(outpipe);
        gst_object_unref(appsink);
        gst_object_unref(inpipe);
        return 1;
    }

    GstCaps* outcaps = gst_caps_new_simple("video/x-raw",
                                           "format", G_TYPE_STRING, "BGR",
                                           "width", G_TYPE_INT, W,
                                           "height", G_TYPE_INT, H,
                                           "framerate", GST_TYPE_FRACTION, FPS_NUM, FPS_DEN,
                                           NULL);
    gst_app_src_set_caps(GST_APP_SRC(appsrc), outcaps);
    gst_caps_unref(outcaps);

    // -------------------------
    // VPI setup (VIC + OFA)
    // -------------------------
    VPIStream stream = nullptr;
    CHECK_VPI(vpiStreamCreate(0, &stream));

    VPIImageWrapperParams wrapParams;
    CHECK_VPI(vpiInitImageWrapperParams(&wrapParams));

    VPIImageData nv12Data{};
    nv12Data.bufferType = VPI_IMAGE_BUFFER_NVBUFFER;
    VPIImage nv12Wrap = nullptr;

    VPIImage cur_y8_bl  = nullptr;
    VPIImage prev_y8_bl = nullptr;
    CHECK_VPI(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER_BL, 0, &cur_y8_bl));
    CHECK_VPI(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER_BL, 0, &prev_y8_bl));

    // For CPU background (pitch-linear)
    VPIImage prev_y8_er = nullptr;
    CHECK_VPI(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER, 0, &prev_y8_er));

    VPIPayload payload = nullptr;
    CHECK_VPI(vpiCreateOpticalFlowDense(VPI_BACKEND_OFA,
                                        W, H,
                                        VPI_IMAGE_FORMAT_Y8_ER_BL,
                                        gridArr, numLevels,
                                        VPI_OPTICAL_FLOW_QUALITY_HIGH,
                                        &payload));

    const int mvW = (W + grid - 1) / grid;
    const int mvH = (H + grid - 1) / grid;

    VPIImage mvImg = nullptr;
    CHECK_VPI(vpiImageCreate(mvW, mvH, VPI_IMAGE_FORMAT_2S16_BL, 0, &mvImg));

    // CPU BGR frame buffer
    const int bgrStride = W * 3;
    std::vector<uint8_t> bgrFrame(size_t(H) * bgrStride);

    bool havePrev = false;
    bool firstCaps = true;

    gst_element_set_state(inpipe, GST_STATE_PLAYING);
    gst_element_set_state(outpipe, GST_STATE_PLAYING);

    // Timestamping
    GstClockTime pts = 0;
    const GstClockTime frameDur = gst_util_uint64_scale_int(1, GST_SECOND, FPS_NUM);

    for (int i = 0; i < 1000000; ++i) {
        GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
        if (!sample) {
            std::cerr << "No sample (EOS or error)\n";
            break;
        }
        if (firstCaps) { print_caps_once(sample); firstCaps = false; }

        GstBuffer* buf = gst_sample_get_buffer(sample);
        if (!buf) { gst_sample_unref(sample); continue; }

        GstMapInfo map;
        NvBufSurface* surf = map_to_nvbufsurface(buf, map);
        if (!surf || surf->numFilled < 1 || !surf->surfaceList) {
            std::cerr << "Failed to access NvBufSurface\n";
            if (surf) gst_buffer_unmap(buf, &map);
            gst_sample_unref(sample);
            continue;
        }

        int fd = static_cast<int>(surf->surfaceList[0].bufferDesc);
        gst_buffer_unmap(buf, &map);
        gst_sample_unref(sample);

        if (fd < 0) continue;

        nv12Data.buffer.fd = fd;

        if (!nv12Wrap) {
            CHECK_VPI(vpiImageCreateWrapper(&nv12Data, &wrapParams, VPI_BACKEND_VIC, &nv12Wrap));
            std::cout << "Initialized NV12 wrapper (fd=" << fd << ")\n";
        } else {
            CHECK_VPI(vpiImageSetWrapper(nv12Wrap, &nv12Data));
        }

        // NV12 -> Y8_ER_BL (VIC)
        CHECK_VPI(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_VIC, nv12Wrap, cur_y8_bl, nullptr));
        CHECK_VPI(vpiStreamSync(stream));

        if (!havePrev) {
            CHECK_VPI(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_VIC, cur_y8_bl, prev_y8_bl, nullptr));
            CHECK_VPI(vpiStreamSync(stream));
            havePrev = true;
            std::cout << "Initialized prev grayscale BL\n";
            continue;
        }

        // OFA flow
        CHECK_VPI(vpiSubmitOpticalFlowDense(stream, VPI_BACKEND_OFA, payload, prev_y8_bl, cur_y8_bl, mvImg));
        CHECK_VPI(vpiStreamSync(stream));

        // Background for overlay: prev_y8_bl -> prev_y8_er
        CHECK_VPI(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_VIC, prev_y8_bl, prev_y8_er, nullptr));
        CHECK_VPI(vpiStreamSync(stream));

        // Lock for CPU access
        VPIImageData prevData;
        CHECK_VPI(vpiImageLockData(prev_y8_er, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &prevData));

        VPIImageData mvData;
        CHECK_VPI(vpiImageLockData(mvImg, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &mvData));

        // --- Build BGR from grayscale background ---
        auto &p0 = prevData.buffer.pitch.planes[0];
        const uint8_t* grayBase = (const uint8_t*)p0.data;
        const int grayPitch = p0.pitchBytes;

        for (int y = 0; y < H; y++) {
            const uint8_t* gy = grayBase + y * grayPitch;
            uint8_t* by = bgrFrame.data() + y * bgrStride;
            for (int x = 0; x < W; x++) {
                uint8_t v = gy[x];
                by[3*x + 0] = v;
                by[3*x + 1] = v;
                by[3*x + 2] = v;
            }
        }

        // --- Draw two vector fields ---
        auto &m0 = mvData.buffer.pitch.planes[0];
        const uint8_t* mvBase = (const uint8_t*)m0.data;
        const int mvPitch = m0.pitchBytes;

        // 1) Fine field (green, thin)
        for (int my = 0; my < mvH; my += fineStrideCells) {
            const int16_t* row = (const int16_t*)(mvBase + my * mvPitch);
            for (int mx = 0; mx < mvW; mx += fineStrideCells) {
                int16_t vx_i = row[2*mx + 0];
                int16_t vy_i = row[2*mx + 1];

                float vx = s10_5_to_px(vx_i);
                float vy = s10_5_to_px(vy_i);

                float mag = std::sqrt(vx*vx + vy*vy);
                if (mag < fineMinMag) continue;

                int x0 = int(mx * grid + grid * 0.5f);
                int y0 = int(my * grid + grid * 0.5f);
                int x1 = int(x0 + vx * fineScale);
                int y1 = int(y0 + vy * fineScale);

                draw_arrow_bgr(bgrFrame.data(), W, H, bgrStride,
                               x0, y0, x1, y1,
                               0, 255, 0, fineThickness); // green
            }
        }

        // 2) Coarse field (red, thick): average of 4x4 motion cells => 16x16 pixels
        const int groupW = mvW / groupCells;  // ignore right edge remainder
        const int groupH = mvH / groupCells;  // ignore bottom remainder

        for (int gy = 0; gy < groupH; gy += coarseStrideGroups) {
            for (int gx = 0; gx < groupW; gx += coarseStrideGroups) {
                float sumVx = 0.0f, sumVy = 0.0f;
                int count = 0;

                const int startY = gy * groupCells;
                const int startX = gx * groupCells;

                for (int oy = 0; oy < groupCells; ++oy) {
                    int my = startY + oy;
                    const int16_t* row = (const int16_t*)(mvBase + my * mvPitch);
                    for (int ox = 0; ox < groupCells; ++ox) {
                        int mx = startX + ox;
                        int16_t vx_i = row[2*mx + 0];
                        int16_t vy_i = row[2*mx + 1];
                        sumVx += s10_5_to_px(vx_i);
                        sumVy += s10_5_to_px(vy_i);
                        count++;
                    }
                }

                float vx = sumVx / float(count);
                float vy = sumVy / float(count);

                float mag = std::sqrt(vx*vx + vy*vy);
                if (mag < coarseMinMag) continue;

                float blockPx = float(groupCells * grid); // 16 px
                int x0 = int((startX * grid) + blockPx * 0.5f);
                int y0 = int((startY * grid) + blockPx * 0.5f);

                int x1 = int(x0 + vx * coarseScale);
                int y1 = int(y0 + vy * coarseScale);

                draw_arrow_bgr(bgrFrame.data(), W, H, bgrStride,
                               x0, y0, x1, y1,
                               0, 0, 255, coarseThickness); // red thick
            }
        }

        CHECK_VPI(vpiImageUnlock(mvImg));
        CHECK_VPI(vpiImageUnlock(prev_y8_er));

        // Push BGR frame to appsrc
        GstBuffer* outbuf = gst_buffer_new_allocate(NULL, bgrFrame.size(), NULL);
        GstMapInfo omap;
        gst_buffer_map(outbuf, &omap, GST_MAP_WRITE);
        std::memcpy(omap.data, bgrFrame.data(), bgrFrame.size());
        gst_buffer_unmap(outbuf, &omap);

        GST_BUFFER_PTS(outbuf) = pts;
        GST_BUFFER_DURATION(outbuf) = frameDur;
        pts += frameDur;

        GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc), outbuf);
        if (ret != GST_FLOW_OK) {
            std::cerr << "appsrc push failed: " << ret << "\n";
            break;
        }

        // Update prev = cur
        CHECK_VPI(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_VIC, cur_y8_bl, prev_y8_bl, nullptr));
        CHECK_VPI(vpiStreamSync(stream));
    }

    gst_app_src_end_of_stream(GST_APP_SRC(appsrc));

    gst_element_set_state(inpipe, GST_STATE_NULL);
    gst_element_set_state(outpipe, GST_STATE_NULL);

    gst_object_unref(appsrc);
    gst_object_unref(outpipe);
    gst_object_unref(appsink);
    gst_object_unref(inpipe);

    if (nv12Wrap)     vpiImageDestroy(nv12Wrap);
    if (cur_y8_bl)    vpiImageDestroy(cur_y8_bl);
    if (prev_y8_bl)   vpiImageDestroy(prev_y8_bl);
    if (prev_y8_er)   vpiImageDestroy(prev_y8_er);
    if (mvImg)        vpiImageDestroy(mvImg);
    if (payload)      vpiPayloadDestroy(payload);
    if (stream)       vpiStreamDestroy(stream);

    return 0;
}
