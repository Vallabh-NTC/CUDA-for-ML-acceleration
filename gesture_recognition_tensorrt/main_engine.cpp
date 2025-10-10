// main_engine.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <memory>
#include <cstring>
#include <cstdlib>      // std::system
#include <opencv2/opencv.hpp>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include "NvInfer.h"

// === Labels del tuo modello (modifica se diverse) ===
static std::vector<std::string> g_labels = {"start", "stop"};

// Logger semplice per TensorRT
struct TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
};

// ---------- Helpers ----------
static void bgr_to_model_input(const cv::Mat& bgr, cv::Mat& out_gray_resized) {
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    int w = gray.cols, h = gray.rows;
    int side = std::min(w, h);
    int x = (w - side) / 2, y = (h - side) / 2;
    cv::Mat square = gray(cv::Rect(x, y, side, side));
    cv::resize(square, out_gray_resized, cv::Size(96, 96), 0, 0, cv::INTER_AREA);
}
static void u8_to_f32(const uint8_t* src, float* dst, int n) {
    const float s = 1.0f / 255.0f;
    for (int i = 0; i < n; ++i) dst[i] = src[i] * s;
}
static void f32_to_f16(const float* src, __half* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = __float2half(src[i]);
}
static std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open engine: " + path);
    f.seekg(0, std::ios::end);
    size_t size = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(size);
    f.read(buf.data(), size);
    return buf;
}
// quoting semplice per shell (singoli apici)
static std::string shell_single_quote(const std::string& s) {
    std::string out; out.reserve(s.size()+8);
    out.push_back('\'');
    for (char c: s) {
        if (c=='\'') out += "'\"'\"'"; else out.push_back(c);
    }
    out.push_back('\'');
    return out;
}
// 1) Replace mqtt_publish with this (absolute path + debug):
static void mqtt_publish(const std::string& host, int port, const std::string& topic, const std::string& payload,
                         const std::string& user="", const std::string& pass="") {
    const char* bin = "/usr/bin/mosquitto_pub"; // absolute path: avoids PATH issues
    std::string cmd = std::string(bin) + " -h " + shell_single_quote(host)
                    + " -p " + std::to_string(port)
                    + " -t " + shell_single_quote(topic)
                    + " -m " + shell_single_quote(payload)
                    + " -d"; // debug from mosquitto_pub
    if (!user.empty()) cmd += " -u " + shell_single_quote(user);
    if (!pass.empty()) cmd += " -P " + shell_single_quote(pass);
    std::cerr << "[MQTT] exec: " << cmd << "\n";
    int rc = std::system(cmd.c_str());
    std::cerr << "[MQTT] rc=" << rc << "\n";
}

// 2) Near the top of main(), right after parsing args, add this test hook:
bool mqtt_test_start = false, mqtt_test_stop = false;
for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--mqtt-test-start") mqtt_test_start = true;
    if (a == "--mqtt-test-stop")  mqtt_test_stop  = true;
}

// If either test flag is set, publish and exit early:
if (mqtt_test_start || mqtt_test_stop) {
    if (mqttHost.empty()) { std::cerr << "[MQTT] --mqtt-host required\n"; return 1; }
    const std::string payload = mqtt_test_start
        ? R"({ "value": { "recording": "start" } })"
        : R"({ "value": { "recording": "stop" } })";
    mqtt_publish(mqttHost, mqttPort, mqttTopic, payload, mqttUser, mqttPass);
    return 0;
}


// ---------- Main ----------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " <engine.plan> [camera_index]\n"
                  << "  " << argv[0] << " <engine.plan> --video <path/to/video.mp4>\n"
                  << "MQTT options:\n"
                  << "  --mqtt-host <ip_or_host>   (required to publish)\n"
                  << "  --mqtt-port <port>         (default 1883)\n"
                  << "  --mqtt-topic <topic>       (default jetson/stream/cmd)\n"
                  << "  --mqtt-user <user>         (optional)\n"
                  << "  --mqtt-pass <pass>         (optional)\n"
                  << "Trigger options:\n"
                  << "  --start-th <p>             (default 0.80)\n"
                  << "  --stop-th  <p>             (default 0.80)\n"
                  << "  --streak <N>               (consecutive frames to trigger, default 3)\n";
        return 1;
    }
    std::string enginePath = argv[1];

    // sorgente input
    bool useVideo = false;
    std::string videoPath;
    int cam_index = 0;

    // MQTT
    std::string mqttHost;  // se vuoto: non pubblica
    int mqttPort = 1883;
    std::string mqttTopic = "jetson/stream/cmd";
    std::string mqttUser, mqttPass;

    // trigger
    float start_th = 0.80f, stop_th = 0.80f;
    int streak_need = 3;

    // parse args
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--video" || a == "-v") && i + 1 < argc) { useVideo = true; videoPath = argv[++i]; }
        else if (a == "--mqtt-host" && i + 1 < argc) mqttHost = argv[++i];
        else if (a == "--mqtt-port" && i + 1 < argc) mqttPort = std::atoi(argv[++i]);
        else if (a == "--mqtt-topic" && i + 1 < argc) mqttTopic = argv[++i];
        else if (a == "--mqtt-user" && i + 1 < argc) mqttUser = argv[++i];
        else if (a == "--mqtt-pass" && i + 1 < argc) mqttPass = argv[++i];
        else if (a == "--start-th" && i + 1 < argc) start_th = std::stof(argv[++i]);
        else if (a == "--stop-th"  && i + 1 < argc) stop_th  = std::stof(argv[++i]);
        else if (a == "--streak"   && i + 1 < argc) streak_need = std::max(1, std::atoi(argv[++i]));
        else if (!useVideo && a.find("--") != 0) { cam_index = std::atoi(a.c_str()); }
    }

    // OpenCV capture
    cv::VideoCapture cap;
    if (useVideo) {
        cap.open(videoPath);
        if (!cap.isOpened()) { std::cerr << "Failed to open video: " << videoPath << std::endl; return 1; }
    } else {
        cap.open(cam_index, cv::CAP_V4L2);
        if (!cap.isOpened()) { std::cerr << "Failed to open camera index " << cam_index << std::endl; return 1; }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);
    }

    // TensorRT init
    TRTLogger logger;
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime) { std::cerr << "createInferRuntime failed\n"; return 1; }
    auto engineData = read_file(enginePath);
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!engine) { std::cerr << "deserializeCudaEngine failed\n"; return 1; }
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context) { std::cerr << "createExecutionContext failed\n"; return 1; }

    const int nbBindings = engine->getNbBindings();
    if (nbBindings < 2) { std::cerr << "Engine needs >= 1 input and 1 output\n"; return 1; }
    int inputIndex = -1, outputIndex = -1;
    for (int i = 0; i < nbBindings; ++i) {
        if (engine->bindingIsInput(i)) inputIndex = i; else outputIndex = i;
    }
    if (inputIndex < 0 || outputIndex < 0) { std::cerr << "Cannot identify bindings\n"; return 1; }

    // Set dims 1x1x96x96 if dynamic
    nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
    if (inputDims.nbDims == 0 || inputDims.d[0] == -1) {
        nvinfer1::Dims4 dims(1, 1, 96, 96);
        if (!context->setBindingDimensions(inputIndex, dims)) { std::cerr << "Failed to set binding dims\n"; return 1; }
        inputDims = context->getBindingDimensions(inputIndex);
    }
    int64_t nInput = 1;
    for (int i = 0; i < inputDims.nbDims; ++i) nInput *= inputDims.d[i];

    // Output dims
    nvinfer1::Dims outputDims = context->getBindingDimensions(outputIndex);
    int64_t nOutput = 1;
    for (int i = 0; i < outputDims.nbDims; ++i) nOutput *= outputDims.d[i];
    if (nOutput <= 0) { std::cerr << "Invalid output size\n"; return 1; }

    // Data types
    nvinfer1::DataType inType = engine->getBindingDataType(inputIndex);
    nvinfer1::DataType outType = engine->getBindingDataType(outputIndex);
    bool inFP16  = (inType  == nvinfer1::DataType::kHALF);
    bool outFP16 = (outType == nvinfer1::DataType::kHALF);

    // Device & host buffers
    void* dInput = nullptr; void* dOutput = nullptr;
    size_t inBytes  = (inFP16 ? sizeof(__half) : sizeof(float)) * nInput;
    size_t outBytes = (outFP16 ? sizeof(__half) : sizeof(float)) * nOutput;
    cudaMalloc(&dInput, inBytes);
    cudaMalloc(&dOutput, outBytes);
    void* bindings[2]; bindings[inputIndex] = dInput; bindings[outputIndex] = dOutput;
    cudaStream_t stream; cudaStreamCreate(&stream);

    std::vector<uint8_t>  h_u8(nInput);
    std::vector<float>    h_f32_in, h_f32_out;
    std::vector<__half>   h_f16_in, h_f16_out;
    if (inFP16)  h_f16_in.resize(nInput);   else h_f32_in.resize(nInput);
    if (outFP16) h_f16_out.resize(nOutput); else h_f32_out.resize(nOutput);

    cv::Mat frame_bgr, gray96;

    // Stato registrazione con debounce
    enum class RecState { IDLE, RECORDING };
    RecState state = RecState::IDLE;
    int start_streak = 0, stop_streak = 0;

    if (mqttHost.empty()) {
        std::cerr << "[MQTT] Warning: --mqtt-host not provided, will NOT publish.\n";
    } else {
        std::cout << "[MQTT] Will publish to " << mqttHost << ":" << mqttPort
                  << " topic=" << mqttTopic << "\n";
    }

    while (true) {
        if (!cap.read(frame_bgr) || frame_bgr.empty()) {
            if (useVideo) { std::cout << "End of stream.\n"; break; }
            else { std::cerr << "Empty frame from camera\n"; continue; }
        }

        // Preprocess → 96x96 uint8
        bgr_to_model_input(frame_bgr, gray96);
        if (!gray96.isContinuous()) gray96 = gray96.clone();
        std::memcpy(h_u8.data(), gray96.data, nInput * sizeof(uint8_t));

        // Normalize + H2D
        if (inFP16) {
            std::vector<float> tmp(nInput);
            u8_to_f32(h_u8.data(), tmp.data(), (int)nInput);
            f32_to_f16(tmp.data(), h_f16_in.data(), (int)nInput);
            cudaMemcpyAsync(dInput, h_f16_in.data(), inBytes, cudaMemcpyHostToDevice, stream);
        } else {
            u8_to_f32(h_u8.data(), h_f32_in.data(), (int)nInput);
            cudaMemcpyAsync(dInput, h_f32_in.data(), inBytes, cudaMemcpyHostToDevice, stream);
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        // Inference
        if (!context->enqueueV2(bindings, stream, nullptr)) { std::cerr << "enqueueV2 failed\n"; break; }

        // D2H
        if (outFP16) {
            cudaMemcpyAsync(h_f16_out.data(), dOutput, outBytes, cudaMemcpyDeviceToHost, stream);
        } else {
            cudaMemcpyAsync(h_f32_out.data(), dOutput, outBytes, cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);

        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // Scores float
        std::vector<float> scores;
        if (outFP16) {
            scores.resize(nOutput);
            for (int i = 0; i < nOutput; ++i) scores[i] = __half2float(h_f16_out[i]);
        } else {
            scores.assign(h_f32_out.begin(), h_f32_out.end());
        }

        // Softmax (commenta se già presente nell'engine)
        auto maxIt = std::max_element(scores.begin(), scores.end());
        float maxv = *maxIt, sum = 0.f;
        for (auto &s : scores) { s = std::exp(s - maxv); sum += s; }
        for (auto &s : scores) s /= sum;

        // Top-1
        int bestIdx = int(std::max_element(scores.begin(), scores.end()) - scores.begin());
        float bestScore = scores[bestIdx];
        std::string bestLab = (bestIdx < (int)g_labels.size()) ? g_labels[bestIdx]
                                                               : ("class_" + std::to_string(bestIdx));
        std::cout << "Pred: " << bestLab << " (" << bestScore << ")  time=" << ms << " ms\n";

        // Debounce + transizioni
        if (bestLab == "start" && bestScore >= start_th) { start_streak++; stop_streak = 0; }
        else if (bestLab == "stop" && bestScore >= stop_th) { stop_streak++; start_streak = 0; }
        else { start_streak = 0; stop_streak = 0; }

        // START
        if (state == RecState::IDLE && start_streak >= streak_need) {
            state = RecState::RECORDING;
            start_streak = 0;
            std::cout << "[TRIGGER] recording START\n";
            if (!mqttHost.empty()) {
                const std::string payload = R"({ "value": { "recording": "start" } })";
                mqtt_publish(mqttHost, mqttPort, mqttTopic, payload, mqttUser, mqttPass);
            }
        }
        // STOP
        if (state == RecState::RECORDING && stop_streak >= streak_need) {
            state = RecState::IDLE;
            stop_streak = 0;
            std::cout << "[TRIGGER] recording STOP\n";
            if (!mqttHost.empty()) {
                const std::string payload = R"({ "value": { "recording": "stop" } })";
                mqtt_publish(mqttHost, mqttPort, mqttTopic, payload, mqttUser, mqttPass);
            }
        }
    }

    cudaStreamDestroy(stream);
    cudaFree(dInput);
    cudaFree(dOutput);
    return 0;
}
