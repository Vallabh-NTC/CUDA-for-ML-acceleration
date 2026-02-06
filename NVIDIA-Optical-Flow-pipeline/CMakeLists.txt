cmake_minimum_required(VERSION 3.18)
project(vpi_of_nvivafilter LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -------------------------
# Sysroot is mandatory
# -------------------------
if(NOT CMAKE_SYSROOT)
  message(FATAL_ERROR "CMAKE_SYSROOT is not set. Use -DCMAKE_SYSROOT=/l4t/targetfs (or your targetfs)")
endif()

# Make CMake search ONLY in sysroot (important for cross builds)
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# -------------------------
# Build target (nvivafilter customer lib)
# -------------------------
add_library(vpi_of SHARED
  src/nvivafilter_vpi_of.cpp
)
set_target_properties(vpi_of PROPERTIES OUTPUT_NAME "vpi_of")

# -------------------------
# Paths
# -------------------------
set(SYS_INC_DIR          ${CMAKE_SYSROOT}/usr/include)
set(SYS_INC_DIR_AARCH64  ${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu)
set(SYS_LIB_DIR          ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)

set(TEGRA_LIB_DIR        ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra)
set(TEGRA_EGL_DIR        ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra-egl)

set(JETSON_MMAPI_INC_DIR ${CMAKE_SYSROOT}/usr/src/jetson_multimedia_api/include)

set(VPI_INC_DIR          ${CMAKE_SYSROOT}/opt/nvidia/vpi2/include)
set(VPI_LIB_DIR          ${CMAKE_SYSROOT}/opt/nvidia/vpi2/lib/aarch64-linux-gnu)
set(VPI_PRIV_LIB_DIR     ${CMAKE_SYSROOT}/opt/nvidia/vpi2/lib/aarch64-linux-gnu/priv)

set(CUPVA_DIR            ${CMAKE_SYSROOT}/opt/nvidia/cupva-2.3)
set(CUPVA_LIB_DIR        ${CUPVA_DIR}/lib/aarch64-linux-gnu)

set(CUDA_INC_DIR         ${CMAKE_SYSROOT}/usr/local/cuda/include)
set(CUDA_LIB_DIR         ${CMAKE_SYSROOT}/usr/local/cuda/lib64)

# -------------------------
# Includes
# -------------------------
target_include_directories(vpi_of PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
  ${SYS_INC_DIR}
  ${SYS_INC_DIR_AARCH64}

  # GStreamer + GLib headers from sysroot
  ${SYS_INC_DIR}/gstreamer-1.0
  ${SYS_INC_DIR}/glib-2.0
  ${SYS_LIB_DIR}/glib-2.0/include

  # Jetson Multimedia API (nvivafilter_customer_api.hpp + NvBufSurface etc.)
  ${JETSON_MMAPI_INC_DIR}

  # VPI + CUDA
  ${VPI_INC_DIR}
  ${CUDA_INC_DIR}
)

# Sanity checks
if(NOT EXISTS "${VPI_INC_DIR}/vpi/VPI.h")
  message(FATAL_ERROR "VPI headers not found: ${VPI_INC_DIR}/vpi/VPI.h")
endif()
# We ship nvivafilter_customer_api.hpp in ./include, so don't warn if MMAPI header is missing in sysroot.
if(EXISTS "${JETSON_MMAPI_INC_DIR}/nvivafilter_customer_api.hpp")
  message(STATUS "Found nvivafilter_customer_api.hpp in MMAPI: ${JETSON_MMAPI_INC_DIR}")
else()
  message(STATUS "Using project-local nvivafilter_customer_api.hpp from ./include")
endif()

if(NOT EXISTS "${CUDA_INC_DIR}/cuda.h")
  message(FATAL_ERROR "CUDA headers not found: ${CUDA_INC_DIR}/cuda.h")
endif()

# -------------------------
# Find libs (sysroot only)
# -------------------------
find_library(GST_LIB            NAMES gstreamer-1.0     HINTS ${SYS_LIB_DIR} REQUIRED)
find_library(GST_BASE_LIB       NAMES gstbase-1.0       HINTS ${SYS_LIB_DIR} REQUIRED)
find_library(GST_APP_LIB        NAMES gstapp-1.0        HINTS ${SYS_LIB_DIR} REQUIRED)
find_library(GST_ALLOCATORS_LIB NAMES gstallocators-1.0 HINTS ${SYS_LIB_DIR} REQUIRED)
find_library(GLIB_LIB           NAMES glib-2.0          HINTS ${SYS_LIB_DIR} REQUIRED)
find_library(GOBJECT_LIB        NAMES gobject-2.0       HINTS ${SYS_LIB_DIR} REQUIRED)

# VPI
find_library(NVVPI_LIB NAMES nvvpi HINTS ${VPI_LIB_DIR} REQUIRED)

# CUPVA (needed because VPI/OFA depends on it on Jetson)
find_library(CUPVA_HOST_LIB
  NAMES cupva_host libcupva_host.so libcupva_host.so.2.3 libcupva_host.so.2.3.0
  HINTS ${CUPVA_LIB_DIR} ${VPI_PRIV_LIB_DIR}
  REQUIRED
)
find_library(CUPVA_HOST_UTILS_LIB
  NAMES cupva_host_utils libcupva_host_utils.so libcupva_host_utils.so.2.3 libcupva_host_utils.so.2.3.0
  HINTS ${CUPVA_LIB_DIR} ${VPI_PRIV_LIB_DIR}
  REQUIRED
)

# PVA deps (commonly required by CUPVA on linkers)
find_library(NVPVA_LIB      NAMES nvpva      HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR} REQUIRED)
find_library(NVPVAINTF_LIB  NAMES nvpvaintf  HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR} REQUIRED)
find_library(NVPVAUMD_LIB   NAMES nvpvaumd   HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR} REQUIRED)

# NvSci deps (optional but helps cross linkers)
find_library(NVSCIBUF_LIB    NAMES nvscibuf     HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR})
find_library(NVSCISYNC_LIB   NAMES nvscisync    HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR})
find_library(NVSCICOMMON_LIB NAMES nvscicommon  HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR})
find_library(NVSCIIPC_LIB    NAMES nvsciipc     HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR})

# EGL for EGLImageKHR wrappers
find_library(EGL_LIB        NAMES EGL        HINTS ${SYS_LIB_DIR} ${TEGRA_EGL_DIR} ${TEGRA_LIB_DIR} REQUIRED)
find_library(EGL_NVIDIA_LIB NAMES EGL_nvidia libEGL_nvidia.so.0 HINTS ${TEGRA_EGL_DIR} ${TEGRA_LIB_DIR})

# CUDA runtime + driver (note: in the .so we mostly need libcuda + libEGL; cudart is optional here,
# but keep it if you later add cudaRuntime calls)
find_library(CUDART_LIB NAMES cudart HINTS ${CUDA_LIB_DIR})
find_library(CUDA_LIB   NAMES cuda   HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR} REQUIRED)

message(STATUS "Using GST:            ${GST_LIB}")
message(STATUS "Using GST_BASE:       ${GST_BASE_LIB}")
message(STATUS "Using GST_APP:        ${GST_APP_LIB}")
message(STATUS "Using GST_ALLOCATORS: ${GST_ALLOCATORS_LIB}")
message(STATUS "Using GLIB:           ${GLIB_LIB}")
message(STATUS "Using GOBJECT:        ${GOBJECT_LIB}")
message(STATUS "Using NVVPI:          ${NVVPI_LIB}")
message(STATUS "Using CUPVA_HOST:     ${CUPVA_HOST_LIB}")
message(STATUS "Using CUPVA_UTILS:    ${CUPVA_HOST_UTILS_LIB}")
message(STATUS "Using EGL:            ${EGL_LIB}")
message(STATUS "Using EGL_NVIDIA:     ${EGL_NVIDIA_LIB}")
message(STATUS "Using CUDA (driver):  ${CUDA_LIB}")
message(STATUS "Using CUDART:         ${CUDART_LIB}")

# -------------------------
# Link libraries
# (order matters for GNU ld in cross builds)
# -------------------------
target_link_libraries(vpi_of PRIVATE
  ${GST_LIB} ${GST_BASE_LIB} ${GST_APP_LIB} ${GST_ALLOCATORS_LIB}
  ${GLIB_LIB} ${GOBJECT_LIB}

  ${NVVPI_LIB}
  ${CUPVA_HOST_LIB}
  ${CUPVA_HOST_UTILS_LIB}

  ${NVPVAINTF_LIB}
  ${NVPVAUMD_LIB}
  ${NVPVA_LIB}

  ${NVSCIBUF_LIB}
  ${NVSCISYNC_LIB}
  ${NVSCICOMMON_LIB}
  ${NVSCIIPC_LIB}

  ${EGL_LIB}
  ${EGL_NVIDIA_LIB}

  ${CUDA_LIB}
  ${CUDART_LIB}

  pthread dl m
)

# -------------------------
# Linker options (cross build)
# -------------------------
target_link_options(vpi_of PRIVATE
  -Wl,--no-as-needed
  -Wl,--copy-dt-needed-entries

  # CUPVA/PVA symbols can be left unresolved for runtime on target
  -Wl,--unresolved-symbols=ignore-in-shared-libs

  -L${SYS_LIB_DIR}
  -L${TEGRA_LIB_DIR}
  -L${TEGRA_EGL_DIR}
  -L${VPI_LIB_DIR}
  -L${VPI_PRIV_LIB_DIR}
  -L${CUPVA_LIB_DIR}
  -L${CUDA_LIB_DIR}

  -Wl,-rpath-link,${SYS_LIB_DIR}
  -Wl,-rpath-link,${TEGRA_LIB_DIR}
  -Wl,-rpath-link,${TEGRA_EGL_DIR}
  -Wl,-rpath-link,${VPI_LIB_DIR}
  -Wl,-rpath-link,${VPI_PRIV_LIB_DIR}
  -Wl,-rpath-link,${CUPVA_LIB_DIR}
  -Wl,-rpath-link,${CUDA_LIB_DIR}

  # Runtime rpath on the Jetson
  -Wl,-rpath,/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:/opt/nvidia/vpi2/lib/aarch64-linux-gnu:/opt/nvidia/vpi2/lib/aarch64-linux-gnu/priv:/opt/nvidia/cupva-2.3/lib/aarch64-linux-gnu:/usr/local/cuda/lib64
)

# -------------------------
# Install path: nvivafilter loads customer libs by absolute path,
# but keeping an install target helps.
# -------------------------
install(TARGETS vpi_of
  LIBRARY DESTINATION usr/local/lib/nvivafilter
)
