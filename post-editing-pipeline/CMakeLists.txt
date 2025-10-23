cmake_minimum_required(VERSION 3.18)

project(post_editing_pipeline LANGUAGES CXX CUDA)

# --- CUDA toolchain root (se serve al cross) ---
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda CACHE PATH "CUDA Toolkit root")
endif()

# Jetson Orin (Ampere): sm_87 — modifica se necessario
set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA architectures")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ---- Sorgenti ----
set(CUDA_SOURCES
    src/color_ops.cu
)

set(CPP_SOURCES
    src/nvivafilter_postediting.cpp
    src/runtime_controls.cpp
)

add_library(nvivafilter_postediting SHARED
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

target_include_directories(nvivafilter_postediting PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /l4t/targetfs/usr/src/jetson_multimedia_api/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/include
    ${CMAKE_SYSROOT}/usr/include
    ${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu
)

# ---- Librerie ----
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

target_link_libraries(nvivafilter_postediting
    ${CUDART_LIBRARY}
    ${CUDA_DRIVER_LIBRARY}
    ${LIB_X11} ${LIB_XEXT} ${LIB_BSD} ${LIB_RT} ${LIB_PTHREAD} ${LIB_M} ${LIB_DL}
)

# ---- Linker search paths (sysroot) ----
target_link_options(nvivafilter_postediting PRIVATE
    -L${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu
    -L${CMAKE_SYSROOT}/lib/aarch64-linux-gnu
    -L${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib
    -L${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib/stubs
)

# ---- Install (.so GStreamer) ----
install(TARGETS nvivafilter_postediting
    LIBRARY DESTINATION lib/aarch64-linux-gnu/gstreamer-1.0/
)
