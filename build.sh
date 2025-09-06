#!/usr/bin/env bash
set -euo pipefail
# Usage:
#   scripts/cmake_docker_build.sh -D BUILD_IMAGE_PIPELINE=ON -D BUILD_HAND_GESTURE=OFF -D CUDA_ARCH=72

# 1) ensure ARM emulation is enabled once (safe to re-run)
if ! docker run --privileged --rm tonistiigi/binfmt --install arm64 >/dev/null 2>&1; then
  echo "binfmt not updated (this is fine)."
fi

# 2) clean old build dir
if [ -d build ]; then
  echo "Removing existing build/ directory..."
  rm -rf build
fi

# 3) run cmake in the ready L4T CUDA image (ARM64; Ubuntu 20.04 / glibc 2.31)
docker run --rm -it --platform linux/arm64 \
  -v "$PWD":/ws -w /ws \
  nvcr.io/nvidia/l4t-cuda:11.4.19-devel \
  bash -lc 'apt-get update && apt-get install -y --no-install-recommends \
              python3-pip make g++ pkg-config \
              libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev && \
            pip3 install --no-cache-dir "cmake>=3.26,<3.29" && \
            /usr/local/bin/cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
              -D BUILD_IMAGE_PIPELINE=ON -D BUILD_HAND_GESTURE=OFF -D CUDA_ARCH=72 && \
            /usr/local/bin/cmake --build build -j$(nproc)'