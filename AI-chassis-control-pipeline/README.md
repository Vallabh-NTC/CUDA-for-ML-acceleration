# Jetson Xavier NX
cmake -S . -B build \
  -DPROJECT=udp-decoder-pipeline \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/Toolchain_aarch64_l4t.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=72
cmake --build build -j"$(nproc)" --verbose

# Jetson Orin AGX
cmake -S . -B build \
  -DPROJECT=udp-decoder-pipeline \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/Toolchain_aarch64_l4t.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j"$(nproc)" --verbose