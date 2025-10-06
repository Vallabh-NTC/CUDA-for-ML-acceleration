# Image Correction Pipeline (CUDA + GStreamer nvivafilter)

## 🎯 Goal

This project implements a **full GPU-based image correction pipeline** for NVIDIA Jetson platforms using `nvivafilter`.  
It provides real-time enhancement of NV12 video streams directly on the GPU, with **zero-copy** processing.  

Main features:
- Fisheye **rectification** (optional).
- **Temporal denoise** (GPU, previous frame blending).
- **Auto exposure (AE)** with anti-flicker smoothing.
- **Auto white balance (AWB)** with gray-world adaptation.
- **Histogram-based tone mapping** and **local CLAHE-like contrast boost**.
- **In-place color grading** (exposure, contrast, saturation).
- **Runtime JSON config** via `ICP_CONTROLS` env var.
- Fully integrated with **GStreamer pipelines** for encoding/streaming.

---

# Jetson Cross-Compilation Environment (JetPack 5.1.1 / L4T R35.3.1)

## 🎯 Goal

Build the nvivafilter CUDA plugin (libnvivafilter_rectify.so) on an x86 host, for Jetson (aarch64), the first time it runs.
This guide gives you copy-pasteable commands and explains why each step matters, so you don’t get bitten by headers, sysroots, or nvcc quirks.

# --- On x86 host --------------------------------------------------------------
export WORKSPACE=$HOME/jetson_ws
mkdir -p "$WORKSPACE"

# Pull official cross-compile container
docker pull nvcr.io/nvidia/jetpack-linux-aarch64-crosscompile-x86:5.1.1

# Start container
docker run -it --rm --privileged --net=host \
  -v /dev/bus/usb:/dev/bus/usb \
  -v "$WORKSPACE":/workspace \
  nvcr.io/nvidia/jetpack-linux-aarch64-crosscompile-x86:5.1.1
bash
Copy code
# --- Inside container ---------------------------------------------------------
# 1) Unpack sysroot + toolchain
cd /l4t
tar -I lbzip2 -xf targetfs.tbz2
mkdir -p toolchain && tar -C toolchain -xf toolchain.tar.gz

cd /workspace
tar -I lbzip2 -xf public_sources.tbz2
cd ./Linux_for_Tegra/source/public
# Jetson Xavier NX
CROSS_COMPILE_AARCH64=/l4t/toolchain/bin/aarch64-buildroot-linux-gnu- CROSS_COMPILE_AARCH64_PATH=/l4t/toolchain NV_TARGET_BOARD=t186ref ./nv_public_src_build.sh
# Jetson Oring AGX
CROSS_COMPILE_AARCH64=/l4t/toolchain/bin/aarch64-buildroot-linux-gnu- \
CROSS_COMPILE_AARCH64_PATH=/l4t/toolchain \
NV_TARGET_BOARD=t234ref \
./nv_public_src_build.sh

cd /l4t/targetfs/usr/src/jetson_multimedia_api
export CROSS_COMPILE=aarch64-linux-gnu-
export TARGET_ROOTFS=/l4t/targetfs/
make

# 2) Fix missing crt startup objects (linker issues)
ln -sf /l4t/targetfs/usr/lib/aarch64-linux-gnu/crt1.o  /l4t/targetfs/lib/crt1.o
ln -sf /l4t/targetfs/usr/lib/aarch64-linux-gnu/crti.o  /l4t/targetfs/lib/crti.o
ln -sf /l4t/targetfs/usr/lib/aarch64-linux-gnu/crtn.o  /l4t/targetfs/lib/crtn.o

# 3) Fix missing glibc include dirs (sys/cdefs.h, bits/wordsize.h)
ln -sfn /l4t/targetfs/usr/include/aarch64-linux-gnu/sys  /l4t/targetfs/usr/include/sys
ln -sfn /l4t/targetfs/usr/include/aarch64-linux-gnu/gnu  /l4t/targetfs/usr/include/gnu
ln -sfn /l4t/targetfs/usr/include/aarch64-linux-gnu/asm  /l4t/targetfs/usr/include/asm

# 4) Fix missing unversioned X11 / libbsd symlinks
ln -sf /l4t/targetfs/usr/lib/aarch64-linux-gnu/libXext.so.6 /l4t/targetfs/usr/lib/aarch64-linux-gnu/libXext.so
ln -sf /l4t/targetfs/usr/lib/aarch64-linux-gnu/libbsd.so.0 /l4t/targetfs/usr/lib/aarch64-linux-gnu/libbsd.so

# 5) Wrap nvcc (strip unsupported flags)
mv /usr/local/cuda/bin/nvcc /usr/local/cuda/bin/nvcc.real
tee /usr/local/cuda/bin/nvcc >/dev/null <<'SH'
#!/usr/bin/env bash
real_nvcc=/usr/local/cuda/bin/nvcc.real
args=(); skip_next=0
for a in "$@"; do
  if [[ $skip_next -eq 1 ]]; then skip_next=0; continue; fi
  case "$a" in
    --target-dir) skip_next=1; continue ;;
    --sysroot=)   continue ;;
    *) args+=("$a") ;;
  esac
done
exec "$real_nvcc" "${args[@]}"
SH
chmod +x /usr/local/cuda/bin/nvcc

# 6) Upgrade CMake (container ships with 3.10, need >=3.18)
apt-get update && apt-get install -y wget build-essential libssl-dev
wget https://cmake.org/files/v3.22/cmake-3.22.6-linux-x86_64.tar.gz
tar -xzf cmake-3.22.6-linux-x86_64.tar.gz -C /usr/local --strip-components=1
cmake --version
bash
Copy code
# --- Build project ------------------------------------------------------------
cd /workspace/CUDA-for-ML-acceleration
rm -rf build

# Jetson Xavier NX
cmake -S . -B build \
  -DPROJECT=image-correction-pipeline \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/Toolchain_aarch64_l4t.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=72
cmake --build build -j"$(nproc)" --verbose

# Jetson Orin AGX
cmake -S . -B build \
  -DPROJECT=image-correction-pipeline \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/Toolchain_aarch64_l4t.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j"$(nproc)" --verbose

# Verify
file build/image-correction-pipeline/libnvivafilter_rectify.so
aarch64-linux-gnu-objdump -p build/image-correction-pipeline/lib

# Deployment on Jetson Xavier

Once the plugin is cross-compiled, copy the shared library to the Jetson Xavier:

scp build/image-correction-pipeline/libnvivafilter_rectify.so jetson@<JETSON_IP>:/usr/local/lib/nvivafilter/

# Transmission Pipeline (Jetson → Host)

Run this on the Jetson Xavier to capture from the CSI camera, apply image correction, encode to H.264, and stream via UDP:

 gst-launch-1.0 -e -v   nvarguscamerasrc ! 'video/x-raw(memory:NVMM),format=NV12,width=1920,height=1080,framerate=30/1'   ! nvivafilter customer-lib-name=/usr/local/lib/nvivafilter/libnvivafilter_imagecorrection.so pre-process=false cuda-process=true post-process=false   ! 'video/x-raw(memory:NVMM),format=NV12'   ! nvv4l2h264enc bitrate=12000000 insert-sps-pps=true idrinterval=30 preset-level=4   ! h264parse ! rtph264pay pt=96 config-interval=1   ! udpsink host=192.168.10.201 port=5000 sync=false async=false

# Receiver pipeline Windows
 
 gst-launch-1.0 -v `
  udpsrc port=5000 caps='application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000' `
  ! rtpjitterbuffer latency=10 do-lost=true `
  ! rtph264depay ! h264parse disable-passthrough=true `
  ! d3d11h264dec `
  ! d3d11convert `
  ! "video/x-raw(memory:D3D11Memory),format=RGBA,colorimetry=bt709,range=tv" `
  ! d3d11videosink sync=false

# Runtime JSON Controls

The filter supports live configuration via a JSON file.
 - Manual exposure in EV: exposure_ev in [-2.0 … +2.0] (negative = darker).
 - Tone controls (clamped to safe ranges):

      - contrast [0.50 … 1.80]
      - highlights [-1.0 … +1.0] (negative = earlier soft roll-off)
      - shadows [-1.0 … +1.0] (positive = lift)
      - whites [-1.0 … +1.0] (negative = lower white ceiling)
      - gamma [0.70 … 1.30]
      - saturation [0.50 … 1.50]
      - tv_range (true = clamp Y to [16..235])

