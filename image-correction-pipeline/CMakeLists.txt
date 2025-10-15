# =============================================================================
# CMake build script for the "image-correction-pipeline" project
#
# Builds a GStreamer "nvivafilter" plugin with CUDA support for Jetson.
#
# Produces:
#   libnvivafilter_imagecorrection.so
# Install path (runtime):
#   /usr/lib/aarch64-linux-gnu/gstreamer-1.0/
# =============================================================================

cmake_minimum_required(VERSION 3.18)

project(image_correction_pipeline LANGUAGES CXX CUDA)

# -----------------------------------------------------------------------------
# CUDA toolkit root (helps cross-builds)
# -----------------------------------------------------------------------------
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda CACHE PATH "CUDA Toolkit root")
endif()

# Jetson Orin (Ampere)
set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA architectures")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# Sources
# -----------------------------------------------------------------------------
set(CUDA_SOURCES
    src/kernel_rectify.cu
    src/wire_lineremoval.cu
    src/color_ops.cu
    src/ei_gesture_infer.cu        # <<< NEW
)

set(CPP_SOURCES
    src/nvivafilter_imagecorrection.cpp
    src/runtime_controls.cpp
)

add_library(nvivafilter_imagecorrection SHARED
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

# -----------------------------------------------------------------------------
# Include directories
# -----------------------------------------------------------------------------
target_include_directories(nvivafilter_imagecorrection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /l4t/targetfs/usr/src/jetson_multimedia_api/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/include
    ${CMAKE_SYSROOT}/usr/include
    ${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu
)

# -----------------------------------------------------------------------------
# Core CUDA libraries
# -----------------------------------------------------------------------------
find_library(CUDART_LIBRARY cudart
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
          ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib
          ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib/stubs
          ${CMAKE_SYSROOT}/usr/local/cuda/lib64
)

find_library(CUDA_DRIVER_LIBRARY cuda
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
          ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib
          ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib/stubs
          ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
)

# -----------------------------------------------------------------------------
# TensorRT (headers + libs) — NO OpenCV
# -----------------------------------------------------------------------------
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include /usr/include/aarch64-linux-gnu /l4t/targetfs/usr/include/aarch64-linux-gnu
)
find_library(TENSORRT_LIB nvinfer
    HINTS /usr/lib /usr/lib/aarch64-linux-gnu /l4t/targetfs/usr/lib/aarch64-linux-gnu
)
find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin
    HINTS /usr/lib /usr/lib/aarch64-linux-gnu /l4t/targetfs/usr/lib/aarch64-linux-gnu
)
if (NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB)
    message(FATAL_ERROR "TensorRT not found. Set TENSORRT_INCLUDE_DIR/TENSORRT_LIB.")
endif()

target_include_directories(nvivafilter_imagecorrection PRIVATE
    ${TENSORRT_INCLUDE_DIR}
)

# -----------------------------------------------------------------------------
# System libs
# -----------------------------------------------------------------------------
find_library(LIB_EGL EGL HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_X11 X11 HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_XEXT Xext HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_BSD bsd HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_RT rt HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_PTHREAD pthread HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_M m HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_DL dl HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)

target_link_libraries(nvivafilter_imagecorrection
    ${CUDART_LIBRARY}
    ${CUDA_DRIVER_LIBRARY}
    ${TENSORRT_LIB}
    ${TENSORRT_PLUGIN_LIB}
    ${LIB_X11}
    ${LIB_XEXT}
    ${LIB_BSD}
    ${LIB_RT}
    ${LIB_PTHREAD}
    ${LIB_M}
    ${LIB_DL}
)

# -----------------------------------------------------------------------------
# Linker search paths for sysroot + CUDA
# -----------------------------------------------------------------------------
target_link_options(nvivafilter_imagecorrection PRIVATE
    -L${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu
    -L${CMAKE_SYSROOT}/lib/aarch64-linux-gnu
    -L${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib
    -L${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib/stubs
)

# Keep RPATH so runtime finds TRT on Jetson
set_target_properties(nvivafilter_imagecorrection PROPERTIES
    BUILD_RPATH "/usr/lib/aarch64-linux-gnu"
    INSTALL_RPATH "/usr/lib/aarch64-linux-gnu"
)

# -----------------------------------------------------------------------------
# Compile options — fix ptxas spillover (-Xptxas=-O3)
# -----------------------------------------------------------------------------
target_compile_options(nvivafilter_imagecorrection PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xptxas=-O3 --expt-relaxed-constexpr>
)

# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------
install(TARGETS nvivafilter_imagecorrection
    LIBRARY DESTINATION lib/aarch64-linux-gnu/gstreamer-1.0/
)
