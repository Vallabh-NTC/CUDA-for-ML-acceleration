#!/usr/bin/env bash
set -euo pipefail
# Usage:
#   scripts/cmake_docker_build.sh -D BUILD_IMAGE_PIPELINE=ON -D BUILD_HAND_GESTURE=OFF -D CUDA_ARCH=72

# 1) ensure ARM emulation is enabled once (safe to re-run)
if ! docker run --privileged --rm tonistiigi/binfmt --install arm64 >/dev/null 2>&1; then
  echo "binfmt not updated (this is fine)."
fi

# 2) run cmake in the ready L4T CUDA image (ARM64; Ubuntu 20.04 / glibc 2.31)
docker run --rm -it --platform linux/arm64 \
  -v "$PWD":/ws -w /ws \
  nvcr.io/nvidia/l4t-cuda:11.4.19-devel \
  bash -lc "apt-get update && apt-get install -y --no-install-recommends \
              cmake make g++ pkg-config \
              libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev && \
            cmake -S . -B build -DCMAKE_BUILD_TYPE=Release $* && \
            cmake --build build -j\$(nproc) && \
            echo && echo 'Built targets:' && (cd build && find . -maxdepth 2 -type f -perm -111 | grep -E '\.\/(image|hand)') || true"
