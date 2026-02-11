rsync -avz ntc-orin@192.168.1.20:/opt/nvidia/vpi2/ /l4t/targetfs/opt/nvidia/vpi2/
rsync -avz --delete   ntc-orin@192.168.1.20:/opt/nvidia/vpi2/include/   /l4t/targetfs/opt/nvidia/vpi2/include/
 
 rsync -avz   ntc-orin@192.168.1.20:/opt/nvidia/cupva-2.3/   /l4t/targetfs/opt/nvidia/cupva-2.3/
 rsync -avz --delete ntc-orin@192.168.1.20:/opt/nvidia/vpi2/lib/aarch64-linux-gnu/  /l4t/targetfs/opt/nvidia/vpi2/lib/aarch64-linux-gnu/

rsync -avz   ntc-orin@192.168.1.20:/opt/nvidia/vpi2/lib/aarch64-linux-gnu/priv/   /l4t/targetfs/opt/nvidia/vpi2/lib/aarch64-linux-gnu/priv/

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


# ------------------------------------------------------------
# VPI OF speed + overlay (Jetson Orin / VPI 2.4) - recommended
# ------------------------------------------------------------

# IMPORTANT: force size if caps are not propagated to the filter
export VPI_OF_W=1280
export VPI_OF_H=720

# Overlay
export VPI_OF_OVERLAY=1
export VPI_OF_OVERLAY_STEP=2
export VPI_OF_OVERLAY_SCALE=3.5

# ROI in MV space (normalized 0..1)
export VPI_OF_ROI_X0=0.3487
export VPI_OF_ROI_X1=0.6615
export VPI_OF_ROI_Y0=0.3823
export VPI_OF_ROI_Y1=0.6049

# Magnitude band-pass (px/frame)
export VPI_OF_MIN_MAG=0.9
export VPI_OF_MAX_MAG=75.0

# Gating / robustness
export VPI_OF_MIN_ROBUST_SAMPLES=64
export VPI_OF_COH_MIN=0.30
export VPI_OF_STD_MAX=12.0

# Tail spike rejection
export VPI_OF_TAIL_RATIO=2.0
export VPI_OF_TAIL_MIN_ABS=35.0

# Calibration (px per meter) -> speed in m/s
export VPI_OF_PX_PER_M=717.0

# EMA smoothing
export VPI_OF_EMA_ALPHA_HI=0.45
export VPI_OF_EMA_ALPHA_LO=0.10

# Optional logging (CSV)
export VPI_OF_LOG=1
export VPI_OF_LOG_PATH=/tmp/vpi_of_speed.csv

# IMU resultant
export VPI_OF_IMU=1
export VPI_OF_IMU_PATH=/home/ntc-orin/Front_and_back_movement_car_test/imu.csv

# optional 
export VPI_OF_IMU_ALPHA_DEG=0.0
export VPI_OF_IMU_G=9.81
export VPI_OF_IMU_AY_BIAS=-2.0
export VPI_OF_IMU_LPF_ALPHA=0.01
export VPI_OF_IMU_OVERLAY_SCALE=40.0

