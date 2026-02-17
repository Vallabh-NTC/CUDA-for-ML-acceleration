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

export VPI_OF_W=1280  #672 for 100FPS
export VPI_OF_H=720   #376 for 100 FPS

# Overlay
export VPI_OF_OVERLAY=1
export VPI_OF_STEP=10
export VPI_OF_OVERLAY_SCALE=3.5

export VPI_OF_ROI_X0=0.35
export VPI_OF_ROI_X1=0.80
export VPI_OF_ROI_Y0=0.70
export VPI_OF_ROI_Y1=0.90

export VPI_OF_FORCE_FPS=100 #or 30,60
export VPI_OF_PX_PER_M=330 #based on the calibration

export VPI_OF_OK_KMH=2
export VPI_OF_SPIKE_KMH=10
export VPI_OF_STABLE_FRAMES=80
export VPI_OF_EMA_TAU=0.60

export VPI_OF_CSV=1
export VPI_OF_CSV_PATH=/home/ntc-orin/Eight_manouver_1/speed_dof.csv   #change the name of the folder based on the specific scenario
export VPI_OF_CSV_EVERY=1

export VPI_OF_TELEM_CSV=/home/ntc-orin/Straight_back_90kmph_100FPS/telemetry.csv
export VPI_OF_TELEM_FRAME_OFFSET=0
export VPI_OF_TELEM_ANCHOR_K=0.05


export VPI_OF_STEER_LUT_CSV=/home/ntc-orin/Eight_manouver_1/steering_local_ratios.csv


gst-launch-1.0 -e filesrc location="/home/ntc-orin/Videos/Eight_manouver_1.mp4" ! qtdemux name=dem dem.video_0 ! queue ! h264parse ! nvv4l2decoder ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12,width=672,height=376' ! nvivafilter cuda-process=true customer-lib-name=/home/ntc-orin/libvpi_of.so ! 'video/x-raw(memory:NVMM),format=NV12' ! nvv4l2h264enc bitrate=4000000 insert-sps-pps=true iframeinterval=30 ! h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host=192.168.1.100 port=5000 sync=false async=false

