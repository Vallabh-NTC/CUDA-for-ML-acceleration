cmake -S . -B build \
  -DPROJECT=NVIDIA-Optical-Flow-pipeline \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/Toolchain_aarch64_l4t.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87

cmake -S . -B build \
  -DPROJECT=NVIDIA-Optical-Flow-pipeline \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/Toolchain_aarch64_l4t.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=72

cmake --build build -j"$(nproc)" --verbose


# RAFT Large Optical Flow Pipeline — Jetson Orin AGX

## Overview

This pipeline estimates vehicle longitudinal velocity (vx) and lateral velocity (vy) in km/h
using RAFT Large optical flow computed on raw camera frames via TensorRT on Jetson Orin AGX.

The processing chain is:
```
NVDEC → EGLImage → NV12→float32 → Sharpen → RAFT Large (TRT) → flow_reduce → FOE correction → Rate Limiter → EMA filter → CSV output
```

---

## Calibrated Parameters

These parameters have been validated against ADMA reference (Sweep_6 maneuver, 100 km/h):

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

# Focus of Expansion (FOE) correction
# Compensates for camera pitch angle (~1.82°) and ROI offset from FOE
# Formula: mean_v_corrected = mean_v - (A * mean_u + B)
export RAFT_FOE_A=-0.0318
export RAFT_FOE_B=0.773

# Exponential Moving Average (EMA) filter windows
# alpha = 2 / (N + 1)
export RAFT_MA_WINDOW=30      # longitudinal velocity vx
export RAFT_MA_WINDOW_V=2     # lateral velocity vy (smaller = more responsive)

# Rate limiter: max allowed change per frame (10ms)
# Physically, 1g acceleration = 0.35 km/h per frame — 2.0 is already very conservative
export RAFT_RATE_LIMIT_KMPH=2.0

# Sharpening strength applied before RAFT inference
export RAFT_SHARP=1.5

# Arrow overlay scale for visualization
export RAFT_RESULT_SCALE=8.0
```

---

## Processing Chain Detail

### 1. FOE Correction
The camera is mounted with a downward pitch of approximately **-1.82°**. This causes all
optical flow vectors to have a spurious downward component even during straight-line driving.

```
mean_v_corrected = mean_v - (A × mean_u + B)
```

- **A = -0.0318** — proportional term, grows with longitudinal speed. Derived from `tan(pitch_angle)`.
- **B = 0.773** — constant offset depending on ROI position relative to the FOE.

At 100 km/h (mean_u ≈ -120 px/frame):
```
correction = -0.0318 × (-120) + 0.773 = 3.82 + 0.773 = 4.59 px/frame
```

> ⚠️ **B must be recalibrated if the ROI is repositioned.** A may need recalibration if the
> camera mounting angle changes or if operating at significantly different speeds from the
> calibration run.

### 2. Rate Limiter
Rejects samples that change faster than physically possible. Applied after FOE correction,
before the EMA filter.

```
if |val - last| > max_delta → clamp to last ± max_delta
```

Maximum physically plausible change at 100 FPS:
```
1g × 0.01s = 0.098 m/s = 0.35 km/h/frame
```

Default limit of 2.0 km/h/frame corresponds to ~5.5g — extremely conservative,
only rejects clear RAFT artifacts.

### 3. EMA Filter
Exponential Moving Average — weights recent samples more heavily than older ones.

```
output[t] = α × input[t] + (1 - α) × output[t-1]
α = 2 / (N + 1)
```

| Parameter          | N  | α     | Latency  | Use                             |
|--------------------|----|-------|----------|---------------------------------|
| RAFT_MA_WINDOW=30  | 30 | 0.064 | ~150ms   | vx — smooth longitudinal        |
| RAFT_MA_WINDOW_V=2 | 2  | 0.667 | ~10ms    | vy — preserve lateral dynamics  |

---

## CSV Output Format

```
frame, mean_u_px, mean_v_px, vx_kmh, vy_kmh
```

| Column      | Description                                                        |
|-------------|--------------------------------------------------------------------|
| frame       | Frame index (starts at 1)                                          |
| mean_u_px   | Mean horizontal optical flow in ROI (px/frame)                     |
| mean_v_px   | Mean vertical optical flow after FOE correction (px/frame)         |
| vx_kmh      | Longitudinal vehicle speed (km/h), rate-limited + EMA filtered     |
| vy_kmh      | Lateral vehicle speed (km/h), rate-limited + EMA filtered          |

**Sign convention:**
- `vx_kmh` positive = vehicle moving forward
- `vy_kmh` positive = vehicle moving to the right

**Side slip angle beta (post-processing):**
```python
beta_deg = np.degrees(np.arctan(vy_kmh / vx_kmh.clip(lower=5)))
```

---

## Velocity Conversion

```
vx_kmh = -mean_u_px × (1 / PX_PER_M) × FPS × 3.6
vy_kmh =  mean_v_px × (1 / PX_PER_M) × FPS × 3.6
```

Where `FPS = 100.0` (hardcoded).

# 200 ms of warmpu
export RAFT_WARMUP_FRAMES=0   # default


---

## Validation Results

### Sweep_6 — 100 km/h constant speed + lateral sweep

| Metric                  | Value   |
|-------------------------|---------|
| vx correlation vs ADMA  | 0.9965  |
| vx error mean           | +0.07 km/h |
| vx error std            | 2.13 km/h  |
| vy offset (straight)    | -0.001 km/h |
| vy correlation vs ADMA  | 0.635   |
| RAFT/ADMA noise ratio   | 1.15×   |

### Log15 — up to 123 km/h + aggressive lateral maneuver

| Metric                  | Value   |
|-------------------------|---------|
| vx correlation vs ADMA  | 0.9970  |
| vx error mean           | +0.39 km/h |
| vx error std            | 2.59 km/h  |
| vy correlation vs ADMA  | 0.9207  |

---

## Recalibration Guide

| Parameter       | When to recalibrate                        | Method                                                        |
|-----------------|--------------------------------------------|---------------------------------------------------------------|
| PX_PER_M        | Camera height or lens changes              | Compare vx mean error vs ADMA at constant speed               |
| FOE_B           | ROI repositioned                           | Measure mean vy offset on straight run vs ADMA                |
| FOE_A           | Camera pitch angle changes                 | Scatter plot mean_v vs mean_u on ADMA-validated straight run  |
| RATE_LIMIT_KMPH | Different FPS or vehicle type              | Set to ~6× max physical acceleration per frame                |

---

## Notes

- Pipeline runs at ~27 FPS with RAFT Large on Jetson Orin AGX (36.7ms inference).
- TensorRT engine: `raft_large_fp16.engine` — I/O is float32 regardless of fp16 flag.
- First frame output is always unreliable (EMA not yet initialized) — skip frame 1 in analysis.
- Beta angle becomes unreliable below ~5 km/h — clip vx to minimum 5 km/h in atan calculation.
- FOE_A has been calibrated at ~100 km/h. At significantly higher speeds (>120 km/h) a small
  residual vx-dependent error on vy may appear — monitor and recalibrate if needed.