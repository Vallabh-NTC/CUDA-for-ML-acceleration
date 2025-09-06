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

# Target system description (always required for cross-compilation).
set(CMAKE_SYSTEM_NAME Linux)          # OS type
set(CMAKE_SYSTEM_PROCESSOR aarch64)   # Target CPU

# Path to Jetson sysroot (extracted targetfs.tbz2 inside container).
set(CMAKE_SYSROOT /l4t/targetfs CACHE PATH "Sysroot for Jetson cross-compilation")

# Cross-compilers from NVIDIA’s L4T toolchain (Buildroot GCC).
set(CMAKE_C_COMPILER   /l4t/toolchain/bin/aarch64-buildroot-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /l4t/toolchain/bin/aarch64-buildroot-linux-gnu-g++)
# CUDA compiler → wrapped nvcc (wrapper strips invalid flags in cross-builds).
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# Tell CMake to search headers/libs inside sysroot, not on host.
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)   # Use host tools
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)    # But target libs from sysroot
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)    # And target includes from sysroot

# Force CUDA compiler detection to succeed without probing target device.
set(CMAKE_CUDA_COMPILER_FORCED TRUE)
set(CMAKE_CUDA_COMPILER_WORKS TRUE)

# Default linker flags for all targets:
# Ensure runtime loader (ld-linux-aarch64.so.1) and glibc are resolved from sysroot.
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-L${CMAKE_SYSROOT}/lib/aarch64-linux-gnu -L${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu")

# Default nvcc flags (forward sysroot include dirs to host g++).
# Needed so nvcc → g++ finds glibc headers (sys/cdefs.h, X11/Xlib.h, etc).
set(CMAKE_CUDA_FLAGS_INIT
    "--compiler-options=-I${CMAKE_SYSROOT}/usr/include \
     --compiler-options=-I${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu \
     --compiler-options=-I${CMAKE_SYSROOT}/usr/include/X11")
