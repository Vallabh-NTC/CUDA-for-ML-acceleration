# =============================================================================
# Toolchain file for cross-compiling to NVIDIA Jetson (aarch64, L4T)
#
# This file tells CMake:
#  - Target CPU architecture
#  - Where to find sysroot
#  - Which compilers to use
#  - How to forward sysroot paths to nvcc
#
# Usage:
#   cmake .. -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/Toolchain_aarch64_l4t.cmake
# =============================================================================

# Target platform
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Sysroot and toolchain
set(CMAKE_SYSROOT "/l4t/targetfs" CACHE PATH "Jetson sysroot")
set(L4T_TOOLCHAIN_ROOT "/l4t/toolchain" CACHE PATH "L4T cross toolchain root")
set(L4T_TRIPLE "aarch64-buildroot-linux-gnu")

# Compilers
set(CMAKE_C_COMPILER   "${L4T_TOOLCHAIN_ROOT}/bin/${L4T_TRIPLE}-gcc")
set(CMAKE_CXX_COMPILER "${L4T_TOOLCHAIN_ROOT}/bin/${L4T_TRIPLE}-g++")

# CUDA compiler (wrapped nvcc)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# IMPORTANT: make nvcc use the cross aarch64 g++ as its host compiler
# CMake will pass this to nvcc as --compiler-bindir
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")

# Ensure CMake resolves headers/libs inside the sysroot
set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)   # host tools
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)    # target libs
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)    # target headers
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Avoid running test binaries during compiler detection
# (try-compile will build a static lib instead of executing)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# CUDA detection in cross builds (don’t probe on target)
set(CMAKE_CUDA_COMPILER_FORCED TRUE)
set(CMAKE_CUDA_COMPILER_WORKS TRUE)

# Help the linker look in the right multiarch spots by default
list(APPEND CMAKE_SYSTEM_LIBRARY_PATH
     "${CMAKE_SYSROOT}/lib/aarch64-linux-gnu"
     "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu")
list(APPEND CMAKE_SYSTEM_INCLUDE_PATH
     "${CMAKE_SYSROOT}/usr/include"
     "${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu")

# Optional: sysroot-aware pkg-config (if present)
# You can also export these in your build env instead.
set(ENV{PKG_CONFIG_SYSROOT_DIR} "${CMAKE_SYSROOT}")
set(ENV{PKG_CONFIG_LIBDIR}      "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig:${CMAKE_SYSROOT}/usr/share/pkgconfig")
set(ENV{PKG_CONFIG_PATH}        "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig:${CMAKE_SYSROOT}/usr/share/pkgconfig")
