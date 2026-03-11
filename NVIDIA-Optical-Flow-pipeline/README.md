# RAFT Large Optical Flow Pipeline — Jetson Orin AGX

## Overview

This pipeline estimates vehicle longitudinal velocity (vx) and lateral velocity (vy) in km/h
using RAFT Large optical flow computed on raw camera frames via TensorRT on Jetson Orin AGX.

Processing chain:
```
NVDEC → EGLImage → NV12→float32 → Sharpen → RAFT Large (TRT) → flow_reduce
→ FOE correction → Startup filter → EMA → CSV + stdout
```

---

## Build

```bash
# Jetson Orin AGX (sm_87)
cmake -S . -B build \
  -DPROJECT=NVIDIA-Optical-Flow-pipeline \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/Toolchain_aarch64_l4t.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87

# Jetson Xavier (sm_72)
cmake -S . -B build \
  -DPROJECT=NVIDIA-Optical-Flow-pipeline \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/Toolchain_aarch64_l4t.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=72

cmake --build build -j"$(nproc)" --verbose
```

---

## Environment Variables

All parameters are configured via environment variables. Set them before launching the
GStreamer pipeline.

```bash
# TensorRT engine path
export RAFT_ENGINE_PATH=/home/ntc-orin/raft/raft_large_fp16.engine

# CSV output path
export RAFT_CSV_PATH=/home/ntc-orin/raft/output.csv

# Scale: pixels per meter in the ROI at current camera mounting distance
export RAFT_PX_PER_M=428

# Region of Interest (normalized [0,1])
export RAFT_ROI_X0=0.62
export RAFT_ROI_X1=0.80
export RAFT_ROI_Y0=0.52
export RAFT_ROI_Y1=0.64
export RAFT_STEP=12

# Focus of Expansion correction
# Formula: mean_v_corrected = mean_v - (A * mean_u + B)
export RAFT_FOE_A=-0.0318
export RAFT_FOE_B=0.773

# Sharpening strength applied before RAFT inference (0 to disable)
export RAFT_SHARP=1.5

# Arrow overlay visual scale
export RAFT_RESULT_SCALE=8.0

# Startup filter
export RAFT_MIN_VX_KMH=5.0       # min plausible vx during startup
export RAFT_MAX_VY_KMH=3.0       # max plausible |vy| during startup
export RAFT_MAX_DVX_KMH=5.0      # max vx change per frame (~14g) during startup
export RAFT_MAX_VALID_GAP=3      # frames before derivative reference expires
export RAFT_MIN_CONSECUTIVE=3    # consecutive valid frames required to lock

# EMA filter (set to 1.0 to disable)
export RAFT_ALPHA_VX=0.4         # vx smoothing — lower = smoother, more delay
export RAFT_ALPHA_VY=0.2         # vy smoothing — lower = smoother, more delay
```

---

## Processing Chain Detail

### 1. FOE Correction

The camera is mounted with a downward pitch of approximately **-1.82°**. This introduces a
spurious downward flow component on all vectors even during straight-line driving.

```
mean_v_corrected = mean_v - (A × mean_u + B)
```

- **A = -0.0318** — proportional term, scales with longitudinal speed (`≈ tan(pitch_angle)`)
- **B = 0.773** — constant offset from ROI position relative to the Focus of Expansion

At 100 km/h (mean_u ≈ -120 px/frame):
```
correction = -0.0318 × (-120) + 0.773 = 4.59 px/frame
```

> ⚠️ **B must be recalibrated if the ROI is repositioned.** A must be recalibrated if the
> camera pitch angle changes.

### 2. Startup Filter

Active only until the pipeline locks onto a stable signal. Once locked, **all frames are
emitted unconditionally** — including during maneuvers where vx may briefly dip below the
startup threshold.

Three gates must all pass for a frame to count toward the lock counter:

**Gate 1 — absolute bounds**
Rejects frames where `vx < RAFT_MIN_VX_KMH` or `|vy| > RAFT_MAX_VY_KMH`. Catches camera
AEC transients at startup and corrupted RAFT inputs (zero-initialized buffer vs first real frame).

**Gate 2 — gap-aware derivative bound**
Rejects frames where `|vx - last_valid_vx| > RAFT_MAX_DVX_KMH`. Only active when the last
valid reference is within `RAFT_MAX_VALID_GAP` frames — beyond that the reference is
considered stale and the gate is skipped, preventing a burst of rejections from permanently
locking out valid data. Default 5 km/h/frame ≈ 14g.

**Gate 3 — consecutive valid frames**
Output is suppressed until `RAFT_MIN_CONSECUTIVE` valid frames are seen in a row. Prevents
isolated transients that pass gates 1 and 2 from being emitted. The derivative reference is
updated throughout, so no stale-reference side effect is introduced.

Once locked, the pipeline logs:
```
[raft_of] LOCKED at frame N after 3 consecutive valid frames
```

### 3. EMA Filter

Exponential Moving Average applied independently to vx and vy after the startup lock.
Initialized at the first post-lock value — no transient at startup.

```
output[t] = α × input[t] + (1 - α) × output[t-1]
```

| Signal | Default α | Approx. delay | Purpose |
|--------|-----------|---------------|---------|
| vx | 0.4 | ~15 ms | Attenuate chassis vibration oscillations |
| vy | 0.2 | ~40 ms | Suppress high-frequency flow noise |

Set `RAFT_ALPHA_VX=1.0` or `RAFT_ALPHA_VY=1.0` to bypass filtering on that channel.

---

## CSV Output Format

```
frame, mean_u_px, mean_v_px, vx_kmh, vy_kmh
```

| Column      | Description                                                    |
|-------------|----------------------------------------------------------------|
| frame       | Frame index                                                    |
| mean_u_px   | Mean raw horizontal flow in ROI (px/frame)                     |
| mean_v_px   | Mean vertical flow after FOE correction (px/frame)             |
| vx_kmh      | Longitudinal speed after EMA filter (km/h)                     |
| vy_kmh      | Lateral speed after EMA filter (km/h)                          |

**Sign convention:**
- `vx_kmh` positive → vehicle moving forward
- `vy_kmh` positive → vehicle moving to the right

**Side slip angle (post-processing):**
```python
beta_deg = np.degrees(np.arctan2(vy_kmh, vx_kmh.clip(lower=5)))
```

---

## Velocity Conversion

```
vx_kmh = -mean_u_px × (1 / PX_PER_M) × FPS × 3.6
vy_kmh =  mean_v_px × (1 / PX_PER_M) × FPS × 3.6
```

`FPS = 100.0` (hardcoded).

---

## Validation Results

### Sweep_6 — 100 km/h constant speed + lateral sweep

| Metric                 | Value       |
|------------------------|-------------|
| vx correlation vs ADMA | 0.9965      |
| vx error mean          | +0.07 km/h  |
| vx error std           | 2.13 km/h   |
| vy offset (straight)   | -0.001 km/h |
| vy correlation vs ADMA | 0.635       |
| RAFT/ADMA noise ratio  | 1.15×       |

### Log15 — up to 123 km/h + aggressive lateral maneuver

| Metric                 | Value      |
|------------------------|------------|
| vx correlation vs ADMA | 0.9970     |
| vx error mean          | +0.39 km/h |
| vx error std           | 2.59 km/h  |
| vy correlation vs ADMA | 0.9207     |

---

## Recalibration Guide

| Parameter  | When to recalibrate               | Method                                                       |
|------------|-----------------------------------|--------------------------------------------------------------|
| PX_PER_M   | Camera height or lens changes     | Compare vx mean error vs ADMA at constant speed              |
| FOE_B      | ROI repositioned                  | Measure mean vy offset on straight run vs ADMA               |
| FOE_A      | Camera pitch angle changes        | Scatter plot mean_v vs mean_u on ADMA-validated straight run |
| ALPHA_VX   | Oscillation amplitude still too high | Lower until amplitude matches ADMA, monitor delay        |
| ALPHA_VY   | HF noise still visible on vy      | Lower toward 0.1; below that latency becomes significant     |

---

## Notes

- Pipeline runs at ~27 FPS with RAFT Large on Jetson Orin AGX (36.7 ms inference).
- TensorRT engine: `raft_large_fp16.engine` — I/O is float32 regardless of fp16 flag.
- Beta angle is unreliable below ~5 km/h — clip vx to minimum 5 km/h in atan2.
- FOE_A calibrated at ~100 km/h. A small residual vx-dependent vy error may appear above
  120 km/h — monitor and recalibrate if needed.
- EMA introduces a causal delay: 15 ms on vx, 40 ms on vy. When computing cross-correlation
  with ADMA, shift the DOF signal forward by the corresponding number of samples.

---

## GStreamer Launch Command

```bash
gst-launch-1.0 -e \
  filesrc location="/home/ntc-orin/Videos/sweep_log_6.mp4" ! \
  qtdemux name=dem dem.video_0 ! queue ! h264parse ! nvv4l2decoder ! \
  nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12,width=672,height=376' ! \
  nvivafilter cuda-process=true \
    customer-lib-name=/home/ntc-orin/libraft_of.so ! \
  'video/x-raw(memory:NVMM),format=NV12' ! \
  nvv4l2h264enc bitrate=4000000 insert-sps-pps=true iframeinterval=30 ! \
  h264parse ! rtph264pay config-interval=1 pt=96 ! \
  udpsink host=192.168.1.10 port=5000 sync=false async=false
```