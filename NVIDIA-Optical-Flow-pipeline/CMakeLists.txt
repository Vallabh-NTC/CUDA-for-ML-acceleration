cmake_minimum_required(VERSION 3.18)
project(raft_of LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── CUDA architecture — Orin AGX = sm_87 ─────────────────────────────────────
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

# ── Sysroot ───────────────────────────────────────────────────────────────────
if(NOT CMAKE_SYSROOT)
  message(FATAL_ERROR "CMAKE_SYSROOT is not set. Pass -DCMAKE_SYSROOT=/l4t/targetfs")
endif()

# ── Paths ─────────────────────────────────────────────────────────────────────
set(SYS_INC          ${CMAKE_SYSROOT}/usr/include)
set(SYS_INC_AARCH64  ${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu)
set(SYS_LIB          ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
set(TEGRA_LIB        ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra)
set(TEGRA_EGL        ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra-egl)

set(CUDA_INC         ${CMAKE_SYSROOT}/usr/local/cuda/include)
set(CUDA_LIB         ${CMAKE_SYSROOT}/usr/local/cuda/lib64)

set(TRT_INC          ${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu)
set(TRT_LIB          ${SYS_LIB})

set(MMAPI_INC        ${CMAKE_SYSROOT}/usr/src/jetson_multimedia_api/include)

set(GST_INC          ${SYS_INC}/gstreamer-1.0)
set(GLIB_INC         ${SYS_INC}/glib-2.0)
set(GLIB_CONFIG_INC  ${SYS_LIB}/glib-2.0/include)

# ── Target ────────────────────────────────────────────────────────────────────
add_library(raft_of SHARED
  src/nvivafilter_raft.cpp
  src/egl_map.cu
  src/nv12_to_rgb_fp16.cu
  src/raft_infer.cu
  src/overlay.cu
  src/flow_reduce.cu
  src/preprocess.cu
)

set_target_properties(raft_of PROPERTIES OUTPUT_NAME "raft_of")

set_source_files_properties(
  src/egl_map.cu
  src/nv12_to_rgb_fp16.cu
  src/raft_infer.cu
  src/overlay.cu
  src/flow_reduce.cu
  src/preprocess.cu
  PROPERTIES LANGUAGE CUDA
)

# ── Includes ──────────────────────────────────────────────────────────────────
target_include_directories(raft_of PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${SYS_INC}
  ${SYS_INC_AARCH64}
  ${CUDA_INC}
  ${TRT_INC}
  ${MMAPI_INC}
  ${GST_INC}
  ${GLIB_INC}
  ${GLIB_CONFIG_INC}
)

# ── Libraries ─────────────────────────────────────────────────────────────────
find_library(LIB_CUDART      NAMES cudart         HINTS ${CUDA_LIB}               REQUIRED)
find_library(LIB_CUDA        NAMES cuda           HINTS ${TEGRA_LIB} ${SYS_LIB}   REQUIRED)
find_library(LIB_EGL         NAMES EGL            HINTS ${TEGRA_EGL} ${TEGRA_LIB} ${SYS_LIB} REQUIRED)
find_library(LIB_NVINFER     NAMES nvinfer        HINTS ${TRT_LIB}                REQUIRED)
find_library(LIB_NVINFPLUGIN NAMES nvinfer_plugin HINTS ${TRT_LIB})
find_library(LIB_GST         NAMES gstreamer-1.0  HINTS ${SYS_LIB}                REQUIRED)
find_library(LIB_GSTBASE     NAMES gstbase-1.0    HINTS ${SYS_LIB}                REQUIRED)
find_library(LIB_GLIB        NAMES glib-2.0       HINTS ${SYS_LIB}                REQUIRED)
find_library(LIB_GOBJECT     NAMES gobject-2.0    HINTS ${SYS_LIB}                REQUIRED)

# ── Compile options ───────────────────────────────────────────────────────────
target_compile_options(raft_of PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    -Xcompiler=-fPIC
  >
  $<$<COMPILE_LANGUAGE:CXX>:
    -O3 -fPIC
  >
)

# ── Link ──────────────────────────────────────────────────────────────────────
target_link_libraries(raft_of PRIVATE
  ${LIB_NVINFER}
  ${LIB_NVINFPLUGIN}
  ${LIB_CUDART}
  ${LIB_CUDA}
  ${LIB_EGL}
  ${LIB_GST}
  ${LIB_GSTBASE}
  ${LIB_GLIB}
  ${LIB_GOBJECT}
  pthread dl m
)

# ── Linker flags / rpath ──────────────────────────────────────────────────────
target_link_options(raft_of PRIVATE
  -Wl,--no-as-needed
  -Wl,--unresolved-symbols=ignore-in-shared-libs
  -L${SYS_LIB}
  -L${TEGRA_LIB}
  -L${TEGRA_EGL}
  -L${CUDA_LIB}
  -L${TRT_LIB}
  -Wl,-rpath-link,${SYS_LIB}
  -Wl,-rpath-link,${TEGRA_LIB}
  -Wl,-rpath-link,${TEGRA_EGL}
  -Wl,-rpath-link,${CUDA_LIB}
  -Wl,-rpath-link,${TRT_LIB}
  -Wl,-rpath,/usr/lib/aarch64-linux-gnu
  -Wl,-rpath,/usr/lib/aarch64-linux-gnu/tegra
  -Wl,-rpath,/usr/lib/aarch64-linux-gnu/tegra-egl
  -Wl,-rpath,/usr/local/cuda/lib64
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS raft_of LIBRARY DESTINATION usr/local/lib/nvivafilter)