# =============================================================================
# CMake build script for the "image-correction-pipeline" project
#
# Builds a GStreamer "nvivafilter" plugin with CUDA support
# for NVIDIA Jetson (cross-compiled from x86 host).
#
# Produces:
#   libnvivafilter_imagecorrection.so
# Install path:
#   /usr/lib/aarch64-linux-gnu/gstreamer-1.0/
#
# Sources:
#   - kernel_rectify.cu  : fisheye undistortion
#   - kernel_color.cu    : color grading, tone mapping, denoise
#   - nvivafilter_imagecorrection.cpp : plugin integration
#   - runtime_controls.cpp : JSON config hot-reload
# =============================================================================


cmake_minimum_required(VERSION 3.18)

project(image_correction_pipeline LANGUAGES CXX CUDA)

# ----------------------------------------------------------------------------- 
# Force CUDA toolkit root (fix cross-build autodetect issues)
# -----------------------------------------------------------------------------
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda CACHE PATH "CUDA Toolkit root")
endif()

# Jetson Xavier/NX/Nano → Volta (sm_72)
set(CMAKE_CUDA_ARCHITECTURES 72 CACHE STRING "CUDA architectures")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ----------------------------------------------------------------------------- 
# Sources
# -----------------------------------------------------------------------------
set(CUDA_SOURCES
    src/kernel_rectify.cu
    src/kernel_color.cu
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
# Libraries
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
    ${LIB_X11}
    ${LIB_XEXT}
    ${LIB_BSD}
    ${LIB_RT}
    ${LIB_PTHREAD}
    ${LIB_M}
    ${LIB_DL}
)

# ----------------------------------------------------------------------------- 
# Sysroot library paths
# -----------------------------------------------------------------------------
target_link_options(nvivafilter_imagecorrection PRIVATE
    -L${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu
    -L${CMAKE_SYSROOT}/lib/aarch64-linux-gnu
    -L${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib
    -L${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib/stubs
)

# ----------------------------------------------------------------------------- 
# Install
# -----------------------------------------------------------------------------
install(TARGETS nvivafilter_imagecorrection
    LIBRARY DESTINATION lib/aarch64-linux-gnu/gstreamer-1.0/
)
