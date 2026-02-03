// vpi_dense_of_nvbuffer.cpp
//
// VPI 2.4.8 - OFA Dense Optical Flow with PYRAMIDS (multi-level)
// Robust version for Jetson that avoids VPI_ERROR_BUFFER_LOCKED by doing stage-by-stage sync.
//
// Build:
// g++ vpi_dense_of_nvbuffer.cpp -o vpi_dense_of_nvbuffer \
//   `pkg-config --cflags --libs opencv4` -lvpi -O3 -std=gnu++17
//
// Run:
// ./vpi_dense_of_nvbuffer /path/to/images 2560 720

#include <vpi/VPI.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/Pyramid.h>

#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/OpticalFlowDense.h>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

#define CHECK_VPI(stmt)                                                         \
    do {                                                                        \
        VPIStatus _st = (stmt);                                                 \
        if (_st != VPI_SUCCESS) {                                               \
            char msg[512];                                                      \
            vpiGetLastStatusMessage(msg, sizeof(msg));                          \
            std::cerr << "VPI error: " << vpiStatusGetName(_st)                 \
                      << " at " << #stmt << "\n"                                \
                      << "Message: " << msg << "\n";                            \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// RAII guard for vpiImageLockData -> guarantees unlock.
struct VPIImageLockGuard {
    VPIImage img = nullptr;
    explicit VPIImageLockGuard(VPIImage i) : img(i) {}
    ~VPIImageLockGuard() { if (img) vpiImageUnlock(img); }
    VPIImageLockGuard(const VPIImageLockGuard&) = delete;
    VPIImageLockGuard& operator=(const VPIImageLockGuard&) = delete;
};

static inline float s10_5_to_px(int16_t v) { return float(v) / 32.0f; }

// Key handling
static bool is_right_key(int k)
{
    return (k == 2555904) || (k == 65363) || (k == 83) || (k == 'd') || (k == 'D');
}
static bool is_left_key(int k)
{
    return (k == 2424832) || (k == 65361) || (k == 81) || (k == 'a') || (k == 'A');
}
static bool is_quit_key(int k)
{
    return (k == 27) || (k == 'q') || (k == 'Q');
}

static std::string make_path(const std::string &dir, int idx)
{
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s/frame_%06d.png", dir.c_str(), idx);
    return std::string(buf);
}

static cv::Mat load_bgr_resized(const std::string &path, int W, int H)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) return cv::Mat();
    if (img.cols != W || img.rows != H)
        cv::resize(img, img, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
    return img;
}

// Upload CV_8UC1 -> VPI Y8_ER pitch-linear
static void upload_y8_to_vpi(VPIImage dstY8, const cv::Mat &gray)
{
    VPIImageData data;
    CHECK_VPI(vpiImageLockData(dstY8, VPI_LOCK_WRITE,
                              VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));
    VPIImageLockGuard guard(dstY8);

    auto &p = data.buffer.pitch.planes[0];
    for (int y = 0; y < gray.rows; ++y) {
        std::memcpy((uint8_t*)p.data + y * p.pitchBytes,
                    gray.ptr<uint8_t>(y),
                    (size_t)gray.cols);
    }
}

// Arrow drawing
static void draw_arrow(cv::Mat &bgr, cv::Point p0, cv::Point p1,
                       const cv::Scalar &color, int thickness)
{
    cv::arrowedLine(bgr, p0, p1, cv::Scalar(0,0,0), thickness+2, cv::LINE_AA, 0, 0.35);
    cv::arrowedLine(bgr, p0, p1, color, thickness, cv::LINE_AA, 0, 0.35);
}

// OpenCV-like arrows: p1 = p0 + flow*scale, clamp maxLen
static void draw_big_green_arrows(cv::Mat& bgr,
                                 const cv::Mat& flowFull, // CV_32FC2
                                 int stepPx,
                                 float scale,
                                 float maxLenPx,
                                 float magThreshold,
                                 int thickness)
{
    for (int y = 0; y < bgr.rows; y += stepPx) {
        for (int x = 0; x < bgr.cols; x += stepPx) {
            const cv::Point2f f = flowFull.at<cv::Point2f>(y, x);
            float mag = std::sqrt(f.x*f.x + f.y*f.y);
            if (mag < magThreshold) continue;

            float dx = f.x * scale;
            float dy = f.y * scale;

            float L = std::sqrt(dx*dx + dy*dy);
            if (L > 1e-6f && L > maxLenPx) {
                float s = maxLenPx / L;
                dx *= s; dy *= s;
            }

            cv::Point p0(x, y);
            cv::Point p1((int)std::lround(x + dx), (int)std::lround(y + dy));
            draw_arrow(bgr, p0, p1, cv::Scalar(0,255,0), thickness);
        }
    }
}

int main(int argc, char** argv)
{
    std::string dir = "/home/ntc-orin/AI_chassis_observer_dataset/images";
    int W = 2560;
    int H = 720;
    if (argc >= 2) dir = argv[1];
    if (argc >= 4) { W = std::atoi(argv[2]); H = std::atoi(argv[3]); }

    // ---- knobs ----
    const int   numLevels = 4;
    const float pyrScale  = 0.5f;   // required for GaussianPyramidGenerator in VPI 2.4.x
    const int   grid      = 4;      // try 8 after it runs stable
    std::vector<int32_t> gridArr(numLevels, grid);

    const bool  USE_CLAHE = true;
    const double INPUT_GAUSS_SIGMA = 0.9;
    const bool  FLIP_Y = false;

    const int   STEP_BIG_PX   = 64;
    const float SCALE_BIG     = 14.0f;
    const float MAXLEN_BIG    = 80.0f;
    const float THRESH_BIG    = 0.05f;
    const int   THICKNESS_BIG = 3;
    const double VIEW_SCALE   = 0.5;

    std::cout << "VPI 2.4.8 OFA Dense OF with PYRAMIDS (multi-level)\n"
              << "Dir: " << dir << "\n"
              << "Size: " << W << "x" << H << "\n"
              << "Levels: " << numLevels << " (scale=" << pyrScale << "), grid=" << grid << "\n"
              << "Keys: Right/d next | Left/a prev | q/ESC quit\n\n";

    // Load first pair
    int idx = 0;
    cv::Mat bgr_prev = load_bgr_resized(make_path(dir, idx), W, H);
    cv::Mat bgr_cur  = load_bgr_resized(make_path(dir, idx + 1), W, H);
    if (bgr_prev.empty() || bgr_cur.empty()) {
        std::cerr << "Need at least 2 frames. Missing: "
                  << make_path(dir, idx) << " or " << make_path(dir, idx+1) << "\n";
        return 1;
    }

    // VPI stream
    VPIStream stream = nullptr;
    CHECK_VPI(vpiStreamCreate(0, &stream));

    // Base images (pitch-linear)
    VPIImage prev_y8_pl = nullptr, cur_y8_pl = nullptr;
    CHECK_VPI(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER, 0, &prev_y8_pl));
    CHECK_VPI(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER, 0, &cur_y8_pl));

    // Pyramids pitch-linear (Y8_ER)
    VPIPyramid prev_pyr_pl = nullptr, cur_pyr_pl = nullptr;
    CHECK_VPI(vpiPyramidCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER, numLevels, pyrScale, 0, &prev_pyr_pl));
    CHECK_VPI(vpiPyramidCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER, numLevels, pyrScale, 0, &cur_pyr_pl));

    // Pyramids block-linear (Y8_ER_BL)
    VPIPyramid prev_pyr_bl = nullptr, cur_pyr_bl = nullptr;
    CHECK_VPI(vpiPyramidCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER_BL, numLevels, pyrScale, 0, &prev_pyr_bl));
    CHECK_VPI(vpiPyramidCreate(W, H, VPI_IMAGE_FORMAT_Y8_ER_BL, numLevels, pyrScale, 0, &cur_pyr_bl));

    // Persistent wrappers
    std::vector<VPIImage> prevLvlPL(numLevels, nullptr), curLvlPL(numLevels, nullptr);
    std::vector<VPIImage> prevLvlBL(numLevels, nullptr), curLvlBL(numLevels, nullptr);
    for (int lvl = 0; lvl < numLevels; ++lvl) {
        CHECK_VPI(vpiImageCreateWrapperPyramidLevel(prev_pyr_pl, lvl, &prevLvlPL[lvl]));
        CHECK_VPI(vpiImageCreateWrapperPyramidLevel(cur_pyr_pl,  lvl, &curLvlPL[lvl]));
        CHECK_VPI(vpiImageCreateWrapperPyramidLevel(prev_pyr_bl, lvl, &prevLvlBL[lvl]));
        CHECK_VPI(vpiImageCreateWrapperPyramidLevel(cur_pyr_bl,  lvl, &curLvlBL[lvl]));
    }

    // OFA payload
    VPIPayload payload = nullptr;
    CHECK_VPI(vpiCreateOpticalFlowDense(VPI_BACKEND_OFA,
                                        W, H,
                                        VPI_IMAGE_FORMAT_Y8_ER_BL,
                                        gridArr.data(), numLevels,
                                        VPI_OPTICAL_FLOW_QUALITY_HIGH,
                                        &payload));

    // Motion vectors
    const int mvW = (W + grid - 1) / grid;
    const int mvH = (H + grid - 1) / grid;

    VPIImage mv_bl = nullptr;
    CHECK_VPI(vpiImageCreate(mvW, mvH, VPI_IMAGE_FORMAT_2S16_BL, 0, &mv_bl));

    VPIImage mv_pl = nullptr;
    CHECK_VPI(vpiImageCreate(mvW, mvH, VPI_IMAGE_FORMAT_2S16, 0, &mv_pl));

    // UI
    cv::namedWindow("OFA pyramids (staged sync)", cv::WINDOW_NORMAL);
    cv::resizeWindow("OFA pyramids (staged sync)", 1280, 360);

    cv::Mat gray_prev, gray_cur;
    cv::Mat mvGrid(mvH, mvW, CV_32FC2);
    cv::Mat flowFull;

    cv::Ptr<cv::CLAHE> clahe;
    if (USE_CLAHE) clahe = cv::createCLAHE(2.0, cv::Size(8,8));

    while (true) {
        // Grayscale
        cv::cvtColor(bgr_prev, gray_prev, cv::COLOR_BGR2GRAY);
        cv::cvtColor(bgr_cur,  gray_cur,  cv::COLOR_BGR2GRAY);

        if (USE_CLAHE) {
            clahe->apply(gray_prev, gray_prev);
            clahe->apply(gray_cur,  gray_cur);
        }
        if (INPUT_GAUSS_SIGMA > 0.0) {
            cv::GaussianBlur(gray_prev, gray_prev, cv::Size(0,0), INPUT_GAUSS_SIGMA);
            cv::GaussianBlur(gray_cur,  gray_cur,  cv::Size(0,0), INPUT_GAUSS_SIGMA);
        }

        // Upload (locks are released before any VPI submit)
        upload_y8_to_vpi(prev_y8_pl, gray_prev);
        upload_y8_to_vpi(cur_y8_pl,  gray_cur);

        // ---------------- Stage 1: build pyramids ----------------
        CHECK_VPI(vpiSubmitGaussianPyramidGenerator(stream, VPI_BACKEND_CPU,
                                                   prev_y8_pl, prev_pyr_pl, VPI_BORDER_CLAMP));
        CHECK_VPI(vpiSubmitGaussianPyramidGenerator(stream, VPI_BACKEND_CPU,
                                                   cur_y8_pl,  cur_pyr_pl,  VPI_BORDER_CLAMP));
        CHECK_VPI(vpiStreamSync(stream));

        // ---------------- Stage 2: PL -> BL per level ----------------
        for (int lvl = 0; lvl < numLevels; ++lvl) {
            CHECK_VPI(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_VIC,
                                                 prevLvlPL[lvl], prevLvlBL[lvl], nullptr));
            CHECK_VPI(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_VIC,
                                                 curLvlPL[lvl],  curLvlBL[lvl],  nullptr));
        }
        CHECK_VPI(vpiStreamSync(stream));

        // ---------------- Stage 3: OFA on pyramids ----------------
        CHECK_VPI(vpiSubmitOpticalFlowDensePyramid(stream, VPI_BACKEND_OFA, payload,
                                                   prev_pyr_bl, cur_pyr_bl, mv_bl));
        CHECK_VPI(vpiStreamSync(stream));

        // ---------------- Stage 4: mv BL -> mv PL ----------------
        CHECK_VPI(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_VIC, mv_bl, mv_pl, nullptr));
        CHECK_VPI(vpiStreamSync(stream));

        // ---------------- Read mv_pl (lock only briefly) ----------------
        {
            VPIImageData mvData;
            CHECK_VPI(vpiImageLockData(mv_pl, VPI_LOCK_READ,
                                       VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &mvData));
            VPIImageLockGuard guard(mv_pl);

            auto &m0 = mvData.buffer.pitch.planes[0];
            const uint8_t *mvBase = (const uint8_t*)m0.data;
            const int mvPitch = m0.pitchBytes;

            for (int my = 0; my < mvH; ++my) {
                const int16_t *row = (const int16_t *)(mvBase + my * mvPitch);
                for (int mx = 0; mx < mvW; ++mx) {
                    float vx = s10_5_to_px(row[2*mx + 0]);
                    float vy = s10_5_to_px(row[2*mx + 1]);
                    if (FLIP_Y) vy = -vy;
                    mvGrid.at<cv::Point2f>(my, mx) = cv::Point2f(vx, vy);
                }
            }
        }

        cv::resize(mvGrid, flowFull, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);

        cv::Mat view = bgr_prev.clone();
        draw_big_green_arrows(view, flowFull, STEP_BIG_PX, SCALE_BIG, MAXLEN_BIG, THRESH_BIG, THICKNESS_BIG);

        cv::Mat small;
        cv::resize(view, small, cv::Size(), VIEW_SCALE, VIEW_SCALE, cv::INTER_AREA);

        cv::setWindowTitle("OFA pyramids (staged sync)",
                           "prev=" + std::to_string(idx) + " cur=" + std::to_string(idx+1) +
                           " | levels=" + std::to_string(numLevels) + " grid=" + std::to_string(grid));
        cv::imshow("OFA pyramids (staged sync)", small);

        int k = cv::waitKeyEx(0);
        if (is_quit_key(k)) break;

        if (is_right_key(k)) idx++;
        else if (is_left_key(k)) idx = std::max(0, idx - 1);
        else continue;

        cv::Mat next_prev = load_bgr_resized(make_path(dir, idx), W, H);
        cv::Mat next_cur  = load_bgr_resized(make_path(dir, idx + 1), W, H);

        if (next_prev.empty()) { std::cerr << "Missing: " << make_path(dir, idx) << "\n"; break; }
        if (next_cur.empty())  { std::cerr << "Missing: " << make_path(dir, idx+1) << "\n"; break; }

        bgr_prev = next_prev;
        bgr_cur  = next_cur;
    }

    // Final sync before destroy
    CHECK_VPI(vpiStreamSync(stream));

    // Destroy wrappers
    for (int lvl = 0; lvl < numLevels; ++lvl) {
        if (prevLvlPL[lvl]) vpiImageDestroy(prevLvlPL[lvl]);
        if (curLvlPL[lvl])  vpiImageDestroy(curLvlPL[lvl]);
        if (prevLvlBL[lvl]) vpiImageDestroy(prevLvlBL[lvl]);
        if (curLvlBL[lvl])  vpiImageDestroy(curLvlBL[lvl]);
    }

    if (mv_pl) vpiImageDestroy(mv_pl);
    if (mv_bl) vpiImageDestroy(mv_bl);
    if (payload) vpiPayloadDestroy(payload);

    if (prev_pyr_bl) vpiPyramidDestroy(prev_pyr_bl);
    if (cur_pyr_bl)  vpiPyramidDestroy(cur_pyr_bl);
    if (prev_pyr_pl) vpiPyramidDestroy(prev_pyr_pl);
    if (cur_pyr_pl)  vpiPyramidDestroy(cur_pyr_pl);

    if (prev_y8_pl) vpiImageDestroy(prev_y8_pl);
    if (cur_y8_pl)  vpiImageDestroy(cur_y8_pl);

    if (stream) vpiStreamDestroy(stream);

    cv::destroyAllWindows();
    return 0;
}
