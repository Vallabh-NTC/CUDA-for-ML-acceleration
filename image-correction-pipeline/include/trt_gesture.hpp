#pragma once

#include <vector>
#include <string>
#include <mutex>
#include <cstdint>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <NvInfer.h>

namespace trt {

struct Engine {
    // --- oggetti TensorRT grezzi ---
    nvinfer1::IRuntime*          runtime  = nullptr;
    nvinfer1::ICudaEngine*       engine   = nullptr;
    nvinfer1::IExecutionContext* context  = nullptr;

    // --- binding info ---
    int inIdx      = -1;
    int outIdx     = -1;
    int nbBindings = 0;    // numero totale di binding dell'engine

    bool inputIsFP16  = false;
    bool outputIsFP16 = false;

    // --- output buffer lato device/host (per l'output "principale") ---
    void*  dOut             = nullptr;   // device output buffer per outIdx
    void*  hostOutPinnedRaw = nullptr;   // pinned host buffer (FP16 o FP32)
    size_t outElems         = 0;         // numero elementi scalari in output tensor (outIdx)

    // altri binding di output hanno i propri buffer device qui
    std::vector<void*> devBindings;      // size = nbBindings; nullptr per gli input, non-null per outputs

    // buffer comodo in float (sempre) per la logica gesture (solo outIdx)
    std::vector<float> hostOut;

    // indice classi nel vettore hostOut
    int idx_start = 0;  // default: 0, override con env GESTURE_IDX_START
    int idx_stop  = 1;  // default: 1, override con env GESTURE_IDX_STOP

    // evento che segnala che l'ultimo memcpy D2H è finito (per outIdx)
    cudaEvent_t ev_trt_done = nullptr;

    // stato interno di commit
    std::mutex mtx;
    bool hasPending   = false;
    bool hasCommitted = false;

    // ------------------------------------------------------------------
    // API pubblica
    // ------------------------------------------------------------------

    // Carica un engine da file e inizializza runtime, engine, context, buffer.
    bool load_from_file(const char* path, cudaStream_t videoStreamForDebug);

    // Rilascia tutte le risorse (da chiamare in destroy_instance)
    void destroy();

    // Controlla se l'evento ev_trt_done è completo e, se sì,
    // copia/converti hostOutPinnedRaw -> hostOut (float).
    // Ritorna true se c'è un nuovo risultato pronto.
    bool try_commit_host_output();

    // Estrae le probabilità per gesture start/stop + top class.
    // sLogit / tLogit: valori grezzi (logit) di start/stop.
    // pStart / pStop: probabilità softmax (su tutto il vettore).
    // top: indice della classe più probabile.
    bool get_start_stop(float& sLogit,
                        float& tLogit,
                        float& pStart,
                        float& pStop,
                        int&   top);

    // Ritorna top-1 class + "probabilità" (qui il valore grezzo hostOut[top]).
    int top1(float* probOut = nullptr) const;
};

} // namespace trt
