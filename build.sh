#!/bin/bash
set -e

# Remove previous build directory
rm -rf build
mkdir -p build
cd build

# Run cmake and build
cmake ..
make -j$(nproc)