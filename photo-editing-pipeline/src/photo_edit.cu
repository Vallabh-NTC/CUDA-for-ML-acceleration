// photo_edit.cu
// Re-process the selected photo whenever editor.json changes.
// Always writes to "<gallery>/<basename>-tmp.jpg" atomically (overwrites the same file on each update).
// The original input image is NEVER modified.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ---- CUDA error helper -------------------------------------------------------
#define CUDA_OK(expr) do {                                     \
    cudaError_t _e = (expr);                                   \
    if (_e != cudaSuccess) {                                   \
        std::fprintf(stderr,                                   \
            "[PHOTO][CUDA] %s:%d: %s -> %s\n",                 \
            __FILE__, __LINE__, #expr, cudaGetErrorString(_e));\
        std::exit(10);                                         \
    }                                                          \
} while(0)

// ---------- tiny helpers ------------------------------------------------------
static inline bool file_exists(const std::string& p) {
    struct stat st; return ::stat(p.c_str(), &st) == 0;
}
static inline time_t file_mtime(const std::string& p) {
    struct stat st; if (::stat(p.c_str(), &st) != 0) return 0; return st.st_mtime;
}
static inline std::string basename_no_ext(const std::string& path) {
    size_t s = path.find_last_of("/\\"); std::string name = (s==std::string::npos)? path : path.substr(s+1);
    size_t dot = name.find_last_of('.'); if (dot==std::string::npos) return name; return name.substr(0,dot);
}
static inline std::string dir_of(const std::string& path) {
    size_t s = path.find_last_of("/\\"); return (s==std::string::npos)? std::string(".") : path.substr(0,s);
}

// ---- atomic file write for "live" preview -----------------------------------
// Write the JPEG to <finalPath>.part, fsync it, rename to <finalPath>, fsync directory.
// This guarantees readers either see the old complete file or the new complete file.
static bool write_jpg_atomic(const std::string& finalPath,
                             int W, int H, const unsigned char* data, int quality)
{
    std::string tmpPath = finalPath + ".part";

    // Write into a temporary file
    if (!stbi_write_jpg(tmpPath.c_str(), W, H, 3, data, quality)) {
        std::fprintf(stderr, "[PHOTO] ERROR: write failed: %s\n", tmpPath.c_str());
        ::unlink(tmpPath.c_str());
        return false;
    }

    // Ensure file contents are flushed to disk
    int tfd = ::open(tmpPath.c_str(), O_RDONLY);
    if (tfd >= 0) { ::fsync(tfd); ::close(tfd); }

    // Atomic replace
    if (::rename(tmpPath.c_str(), finalPath.c_str()) != 0) {
        std::fprintf(stderr, "[PHOTO] ERROR: rename failed %s -> %s (%s)\n",
                     tmpPath.c_str(), finalPath.c_str(), std::strerror(errno));
        ::unlink(tmpPath.c_str());
        return false;
    }

    // Flush the directory metadata so the rename is durable/visible immediately
    std::string dir = dir_of(finalPath);
    int dfd = ::open(dir.c_str(),
#ifdef O_DIRECTORY
                     O_RDONLY | O_DIRECTORY
#else
                     O_RDONLY
#endif
    );
    if (dfd >= 0) { ::fsync(dfd); ::close(dfd); }

    return true;
}

// -------- params (subset aligned with your editor.json keys) ------------------
struct Params {
    bool   enable      = true;
    float  contrast    = 1.00f;  // 0.50 .. 1.80
    float  brightness  = 0.00f;  // -1.00 .. +1.00 (adds after [0..1] mapping)
    float  gamma       = 1.00f;  // 0.50 .. 2.00
    float  saturation  = 1.00f;  // 0.00 .. 4.00
};

// Very tolerant JSON puller (like your ColorConfigIO)
static float pullFloat(const std::string& s, const char* key, float def) {
    std::string pat = std::string("\"") + key + "\"";
    size_t k = s.find(pat); if (k==std::string::npos) return def;
    k = s.find(':', k); if (k==std::string::npos) return def;
    size_t e = s.find_first_of(",}\n\r", k+1);
    std::string num = s.substr(k+1, (e==std::string::npos?s.size():e)-(k+1));
    try { return std::stof(num); } catch(...) { return def; }
}
static bool pullBool(const std::string& s, const char* key, bool def) {
    std::string pat = std::string("\"") + key + "\"";
    size_t k = s.find(pat); if (k==std::string::npos) return def;
    k = s.find(':', k); if (k==std::string::npos) return def;
    std::string val = s.substr(k+1, 8);
    return val.find("true") != std::string::npos;
}

static Params read_params_from_editor_json(const std::string& jsonPath, const Params& prev) {
    FILE* f = ::fopen(jsonPath.c_str(), "rb");
    if (!f) return prev;
    std::string data;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf,1,sizeof(buf),f))>0) data.append(buf,n);
    std::fclose(f);

    Params p = prev;
    p.enable     = pullBool (data, "enable",     p.enable);
    p.contrast   = pullFloat(data, "contrast",   p.contrast);
    p.brightness = pullFloat(data, "brightness", p.brightness);
    p.gamma      = pullFloat(data, "gamma",      p.gamma);
    p.saturation = pullFloat(data, "saturation", p.saturation);

    // Clamp to runtime ranges
    auto clamp = [](float v, float lo, float hi){ return v<lo?lo:(v>hi?hi:v); };
    p.contrast   = clamp(p.contrast,   0.50f, 1.80f);
    p.brightness = clamp(p.brightness, -1.0f, 1.0f);
    p.gamma      = clamp(p.gamma,       0.5f, 2.0f);
    p.saturation = clamp(p.saturation,  0.0f, 4.0f);

    return p;
}

// -------- CUDA bits (simple RGB kernel) --------------------------------------
__device__ inline float clampf(float v, float a, float b){ return v<a?a:(v>b?b:v); }

__device__ inline float3 rgb2hsv(float3 c) {
    float mx = fmaxf(c.x, fmaxf(c.y, c.z));
    float mn = fminf(c.x, fminf(c.y, c.z));
    float d  = mx - mn;
    float h = 0.f;
    if (d > 1e-6f) {
        if (mx == c.x)      h = fmodf(((c.y - c.z) / d), 6.f);
        else if (mx == c.y) h = ((c.z - c.x) / d) + 2.f;
        else                h = ((c.x - c.y) / d) + 4.f;
        h *= 60.f; if (h < 0.f) h += 360.f;
    }
    float s = (mx <= 0.f) ? 0.f : (d / mx);
    float v = mx;
    return make_float3(h,s,v);
}
__device__ inline float3 hsv2rgb(float3 h) {
    float H=h.x, S=h.y, V=h.z;
    if (S<=1e-6f) return make_float3(V,V,V);
    float C = V*S;
    float X = C*(1.f - fabsf(fmodf(H/60.f,2.f)-1.f));
    float m = V - C;
    float3 r;
    if      (H<60)   r=make_float3(C,X,0);
    else if (H<120)  r=make_float3(X,C,0);
    else if (H<180)  r=make_float3(0,C,X);
    else if (H<240)  r=make_float3(0,X,C);
    else if (H<300)  r=make_float3(X,0,C);
    else             r=make_float3(C,0,X);
    r.x += m; r.y += m; r.z += m;
    return r;
}

__global__ void apply_params_rgb(
    const unsigned char* __restrict__ in,
    unsigned char* __restrict__ out,
    int W, int H, int stride,
    float contrast, float brightness, float gamma, float saturation)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x>=W || y>=H) return;

    const unsigned char* src = in + y*stride + 3*x;
    unsigned char*       dst = out + y*stride + 3*x;

    float r = src[0]/255.f, g = src[1]/255.f, b = src[2]/255.f;

    // Contrast around mid-gray 0.5
    r = (r - 0.5f) * contrast + 0.5f;
    g = (g - 0.5f) * contrast + 0.5f;
    b = (b - 0.5f) * contrast + 0.5f;

    // Global gamma
    float invg = (gamma>1e-6f) ? (1.0f/gamma) : 1.f;
    r = powf(clampf(r,0,1), invg);
    g = powf(clampf(g,0,1), invg);
    b = powf(clampf(b,0,1), invg);

    // Brightness (full-range offset)
    r = clampf(r + brightness, 0.f, 1.f);
    g = clampf(g + brightness, 0.f, 1.f);
    b = clampf(b + brightness, 0.f, 1.f);

    // Saturation in HSV
    float3 hsv = rgb2hsv(make_float3(r,g,b));
    hsv.y *= saturation;
    hsv.y = clampf(hsv.y, 0.f, 4.f);
    float3 rr = hsv2rgb(hsv);

    dst[0] = (unsigned char)(clampf(rr.x,0.f,1.f)*255.f + 0.5f);
    dst[1] = (unsigned char)(clampf(rr.y,0.f,1.f)*255.f + 0.5f);
    dst[2] = (unsigned char)(clampf(rr.z,0.f,1.f)*255.f + 0.5f);
}

// -------- main loop -----------------------------------------------------------
int main(int argc, char** argv) {
    std::string inPath, galleryDir, editorJson = "/home/moviemaker/editor.json";

    // Args passed from jetson_editor
    for (int i=1;i<argc;i++) {
        std::string a = argv[i];
        auto need = [&](const char* flag){
            if (a==flag && i+1<argc){ return true; } return false;
        };
        if (need("--input"))       { inPath = argv[++i]; continue; }
        if (need("--gallery"))     { galleryDir = argv[++i]; continue; }
        if (need("--editor-json")) { editorJson = argv[++i]; continue; }
        if (a=="--help"||a=="-h"){
            std::fprintf(stderr,
                "Usage: %s --input <img> --gallery <dir> [--editor-json <path>]\n", argv[0]);
            return 1;
        }
    }

    if (inPath.empty()) {
        std::fprintf(stderr, "[PHOTO] ERROR: --input is required\n");
        return 2;
    }
    if (galleryDir.empty()) galleryDir = dir_of(inPath);

    const std::string base   = basename_no_ext(inPath);
    const std::string outTmp = galleryDir + "/" + base + "-tmp.jpg";

    std::printf("[PHOTO] Photo editing started\n");
    std::printf("[PHOTO] Selected file: %s\n", base.c_str());
    std::printf("[PHOTO] Full path:     %s\n", inPath.c_str());
    std::printf("[PHOTO] Output (live tmp): %s\n", outTmp.c_str());
    std::printf("[PHOTO] Watching editor.json: %s\n", editorJson.c_str());

    // Load original once (RGB8). The original file is NEVER modified.
    int W=0,H=0,N=0;
    stbi_uc* img = stbi_load(inPath.c_str(), &W, &H, &N, 3);
    if (!img) {
        std::fprintf(stderr, "[PHOTO] ERROR: failed to load image: %s\n", inPath.c_str());
        return 3;
    }
    const size_t stride = (size_t)W * 3;
    const size_t bytes  = (size_t)H * stride;

    // Device buffers (kept alive for all passes)
    unsigned char *d_in=nullptr, *d_out=nullptr;
    CUDA_OK(cudaMalloc(&d_in,  bytes));
    CUDA_OK(cudaMalloc(&d_out, bytes));
    CUDA_OK(cudaMemcpy(d_in, img, bytes, cudaMemcpyHostToDevice));

    // Defaults (will be overwritten by json on first pass if present)
    Params params;
    time_t last_mtime = 0;

    auto run_once = [&](const Params& p){
        dim3 b(16,16), g((W+b.x-1)/b.x,(H+b.y-1)/b.y);
        apply_params_rgb<<<g,b>>>(d_in, d_out, W, H, (int)stride,
                                  p.contrast, p.brightness, p.gamma, p.saturation);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());

        // Use pinned host memory for faster D->H (optional micro-optimization)
        unsigned char* host = nullptr;
        CUDA_OK(cudaMallocHost(&host, bytes));
        CUDA_OK(cudaMemcpy(host, d_out, bytes, cudaMemcpyDeviceToHost));

        // Atomic write to a fixed "-tmp.jpg" file
        if (write_jpg_atomic(outTmp, W, H, host, 90)) {
            std::printf("[PHOTO] wrote: %s  (contrast=%.3f, bright=%.3f, gamma=%.3f, sat=%.3f)\n",
                        outTmp.c_str(), p.contrast, p.brightness, p.gamma, p.saturation);
        }

        CUDA_OK(cudaFreeHost(host));
    };

    // Force an initial run (even if editor.json is missing) → ensures tmp file exists immediately
    run_once(params);

    // Watch loop: when editor.json mtime changes, re-run and overwrite the SAME tmp file
    while (true) {
        time_t mt = file_mtime(editorJson);
        if (mt != 0 && mt != last_mtime) {
            last_mtime = mt;
            Params p2 = read_params_from_editor_json(editorJson, params);
            params = p2;
            if (params.enable) {
                run_once(params);
            } else {
                std::printf("[PHOTO] enable=false -> skipping write\n");
            }
        }
        // Keep it light; the parent process will kill us on "stop"
        usleep(150 * 1000); // 150 ms
    }

    // Never reached in normal flow
    CUDA_OK(cudaFree(d_in)); CUDA_OK(cudaFree(d_out));
    stbi_image_free(img);
    return 0;
}
