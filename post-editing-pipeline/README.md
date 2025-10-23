# --- Build project ------------------------------------------------------------
cd /workspace/CUDA-for-ML-acceleration
rm -rf build

# Jetson Xavier NX
cmake -S . -B build \
  -DPROJECT=post-editing-pipeline \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/Toolchain_aarch64_l4t.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=72
cmake --build build -j"$(nproc)" --verbose

# Jetson Orin AGX
cmake -S . -B build \
  -DPROJECT=post-editing-pipeline \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/Toolchain_aarch64_l4t.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j"$(nproc)" --verbose

# Verify
file build/image-correction-pipeline/libnvivafilter_rectify.so
aarch64-linux-gnu-objdump -p build/image-correction-pipeline/lib

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

