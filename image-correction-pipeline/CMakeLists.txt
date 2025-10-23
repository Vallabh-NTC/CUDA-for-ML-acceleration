# =============================================================================
# CMake build script for the "image-correction-pipeline" project
#
# Builds a GStreamer "nvivafilter" plugin with CUDA + TensorRT on NVIDIA Jetson
# (cross-compiled from x86 host using an L4T sysroot).
#
# Output:
#   libnvivafilter_imagecorrection.so
# Install path:
#   /usr/lib/aarch64-linux-gnu/gstreamer-1.0/
# =============================================================================

cmake_minimum_required(VERSION 3.18)

project(image_correction_pipeline LANGUAGES CXX CUDA)

# -----------------------------------------------------------------------------
# CUDA toolchain
# -----------------------------------------------------------------------------
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda CACHE PATH "CUDA Toolkit root")
endif()

# Jetson Orin AGX -> Ampere (sm_87)
set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA architectures")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# Sources
# -----------------------------------------------------------------------------
set(CUDA_SOURCES
    src/kernel_rectify.cu
    src/wire_lineremoval.cu
    src/color_ops.cu
    src/ei_gesture_infer.cu
)

set(CPP_SOURCES
    src/nvivafilter_imagecorrection.cpp
    src/runtime_controls.cpp
    src/trt_gesture.cpp
)

add_library(nvivafilter_imagecorrection SHARED
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

# -----------------------------------------------------------------------------
# Include directories (order matters: sysroot first to match target versions)
# -----------------------------------------------------------------------------
# NOTE: adjust /l4t/targetfs to your actual sysroot path if different.
target_include_directories(nvivafilter_imagecorrection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include

    # Jetson/L4T sysroot headers (match target versions)
    ${CMAKE_SYSROOT}/usr/include
    ${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu
    /l4t/targetfs/usr/include/aarch64-linux-gnu
    /l4t/targetfs/usr/src/jetson_multimedia_api/include

    # CUDA headers (toolkit + target)
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/include
)

# -----------------------------------------------------------------------------
# Libraries
# -----------------------------------------------------------------------------
# Do NOT use 'stubs' for final link; link real device libs from sysroot.
find_library(CUDART_LIBRARY cudart
    HINTS
        ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CMAKE_SYSROOT}/usr/local/cuda/lib64
)

find_library(CUDA_DRIVER_LIBRARY cuda
    HINTS
        ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
        ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

# TensorRT runtime (REQUIRED)
find_library(LIB_NVINFER nvinfer
    HINTS
        ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu
        ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
        /l4t/targetfs/usr/lib/aarch64-linux-gnu
        /l4t/targetfs/usr/lib/aarch64-linux-gnu/tegra
        /usr/lib/aarch64-linux-gnu
        /usr/lib/aarch64-linux-gnu/tegra
)
if (NOT LIB_NVINFER)
    message(FATAL_ERROR "libnvinfer.so not found in sysroot. Install TensorRT runtime in the target sysroot.")
endif()

find_library(LIB_EGL EGL HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_X11 X11 HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_XEXT Xext HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_BSD bsd HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_RT rt HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_PTHREAD pthread HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_M m HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_DL dl HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)

target_link_libraries(nvivafilter_imagecorrection
    ${LIB_NVINFER}            # <-- TensorRT runtime
    ${CUDART_LIBRARY}
    ${CUDA_DRIVER_LIBRARY}
    ${LIB_X11}
    ${LIB_XEXT}
    ${LIB_BSD}
    ${LIB_RT}
    ${LIB_PTHREAD}
    ${LIB_M}
    ${LIB_DL}
)

# -----------------------------------------------------------------------------
# Linker search paths for cross link (not used at runtime)
# -----------------------------------------------------------------------------
target_link_options(nvivafilter_imagecorrection PRIVATE
    -Wl,--no-as-needed
    -L${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu
    -L${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
    -L${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib
)

# -----------------------------------------------------------------------------
# Runtime RPATH on the target device (so loader finds TRT & CUDA)
# -----------------------------------------------------------------------------
set_target_properties(nvivafilter_imagecorrection PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra"
)

# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------
install(TARGETS nvivafilter_imagecorrection
    LIBRARY DESTINATION lib/aarch64-linux-gnu/gstreamer-1.0/
)

# -----------------------------------------------------------------------------
# Helpful diagnostics
# -----------------------------------------------------------------------------
message(STATUS "CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CMAKE_SYSROOT          = ${CMAKE_SYSROOT}")
message(STATUS "Found libnvinfer: ${LIB_NVINFER}")
message(STATUS "Found cudart:      ${CUDART_LIBRARY}")
message(STATUS "Found cuda (drv):  ${CUDA_DRIVER_LIBRARY}")
