rsync -avz ntc-orin@192.168.1.10:/opt/nvidia/vpi2/ /l4t/targetfs/opt/nvidia/vpi2/
rsync -avz --delete   ntc-orin@192.168.1.10:/opt/nvidia/vpi2/include/   /l4t/targetfs/opt/nvidia/vpi2/include/
 
 rsync -avz   ntc-orin@192.168.1.10:/opt/nvidia/cupva-2.3/   /l4t/targetfs/opt/nvidia/cupva-2.3/
 rsync -avz --delete ntc-orin@192.168.1.10:/opt/nvidia/vpi2/lib/aarch64-linux-gnu/  /l4t/targetfs/opt/nvidia/vpi2/lib/aarch64-linux-gnu/

rsync -avz   ntc-orin@192.168.1.10:/opt/nvidia/vpi2/lib/aarch64-linux-gnu/priv/   /l4t/targetfs/opt/nvidia/vpi2/lib/aarch64-linux-gnu/priv/

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

export VPI_OF_ROI_X0=0.3487
export VPI_OF_ROI_X1=0.6615
export VPI_OF_ROI_Y0=0.3823
export VPI_OF_ROI_Y1=0.6049


export VPI_OF_OVERLAY_STEP=2


## Strong robustness against shadow-induced random vectors
# Robust gating
export VPI_OF_COH_MIN=0.60
export VPI_OF_STD_MAX=6.0
export VPI_OF_MIN_ROBUST_SAMPLES=160

# Tail spike rejection
export VPI_OF_TAIL_RATIO=2.0
export VPI_OF_TAIL_MIN_ABS=35.0

# Magnitude band-pass
export VPI_OF_MIN_MAG=0.9
export VPI_OF_MAX_MAG=75.0

# Luma gates
export VPI_OF_GRAD_MIN=20
export VPI_OF_DY_MAX=26

# NEW: direction + trimming + accel limiter
export VPI_OF_DIR_COS_MIN=0.78     # 0.70..0.85 typical
export VPI_OF_TRIM_K=2.5           # 2.0..3.0 typical
export VPI_OF_MAX_ACCEL=10.0       # 6..15 typical

# EMA
export VPI_OF_EMA_ALPHA_HI=0.45
export VPI_OF_EMA_ALPHA_LO=0.10


export VPI_OF_FORBID=1
export VPI_OF_FORBID_MLOW=-0.30
export VPI_OF_FORBID_MHIGH=+0.30


