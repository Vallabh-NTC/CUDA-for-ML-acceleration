rsync -avz ntc-orin@192.168.1.10:/opt/nvidia/vpi2/ /l4t/targetfs/opt/nvidia/vpi2/
rsync -avz --delete   ntc-orin@192.168.1.10:/opt/nvidia/vpi2/include/   /l4t/targetfs/opt/nvidia/vpi2/include/
 
 rsync -avz   ntc-orin@192.168.1.10:/opt/nvidia/cupva-2.3/   /l4t/targetfs/opt/nvidia/cupva-2.3/
 rsync -avz --delete \
>   ntc-orin@192.168.1.10:/opt/nvidia/vpi2/lib/aarch64-linux-gnu/ \
>   /l4t/targetfs/opt/nvidia/vpi2/lib/aarch64-linux-gnu/

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