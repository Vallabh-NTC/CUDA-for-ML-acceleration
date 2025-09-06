# =============================================================================
# CMake build script for the "image-correction-pipeline" project
#
# Builds a GStreamer "nvivafilter" plugin with CUDA support
# for NVIDIA Jetson (cross-compiled from x86 host).
# =============================================================================

cmake_minimum_required(VERSION 3.18)

project(image_correction_pipeline LANGUAGES CXX CUDA)

# Jetson Xavier/NX/Nano → GPU arch "72" (Volta, sm_72).
# Explicitly set to avoid CMake error "CUDA_ARCHITECTURES is empty".
set(CMAKE_CUDA_ARCHITECTURES 72 CACHE STRING "CUDA architectures")

# Enforce C++17 and CUDA C++17 dialects.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CUDA source files (compiled with nvcc → kernels + host glue).
set(CUDA_SOURCES
    src/nvivafilter_rectify.cu
    src/rectify_kernels.cu
)

# Host C++ sources (compiled with aarch64 g++).
set(CPP_SOURCES
    src/runtime_controls.cpp
)

# Build a shared library (GStreamer plugin).
# Name must match GStreamer conventions → "libnvivafilter_rectify.so"
add_library(nvivafilter_rectify SHARED
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

# Tell CMake to treat CUDA sources as CUDA (not plain C++).
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

# Include dirs for both host compiler and nvcc.
# - Our own headers under include/
# - CUDA Toolkit headers
# - Sysroot headers (glibc, EGL, X11, Jetson Multimedia API)
target_include_directories(nvivafilter_rectify PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${CMAKE_SYSROOT}/usr/include
    ${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu
)

# Find CUDA runtime library (cudart) inside sysroot/host toolkit.
# Note: search order is important → first stubs, then real libs.
find_library(CUDART_LIBRARY cudart
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
          ${CUDA_TOOLKIT_ROOT_DIR}/lib/stubs
          ${CMAKE_SYSROOT}/usr/local/cuda/lib64
)

# Link cudart if found (avoids hardcoding -lcudart).
if (CUDART_LIBRARY)
    target_link_libraries(nvivafilter_rectify ${CUDART_LIBRARY})
endif()

# Add sysroot library search paths at link time.
# Without this step, ld would try to resolve libc/ld-linux from host /lib64.
target_link_options(nvivafilter_rectify PRIVATE
    -L${CMAKE_SYSROOT}/lib/aarch64-linux-gnu
    -L${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu
)

# Install plugin into GStreamer standard directory inside sysroot.
# On Jetson this resolves to: /usr/lib/aarch64-linux-gnu/gstreamer-1.0/
install(TARGETS nvivafilter_rectify
    LIBRARY DESTINATION lib/aarch64-linux-gnu/gstreamer-1.0/
)
