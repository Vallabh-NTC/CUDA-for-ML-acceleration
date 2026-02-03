cmake_minimum_required(VERSION 3.18)
project(vpi_dense_of_nvbuffer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(vpi_dense_of_nvbuffer
  src/vpi_dense_of_nvbuffer.cpp
)

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
target_include_directories(vpi_dense_of_nvbuffer PRIVATE
  ${SYS_INC_DIR}
  ${SYS_INC_DIR_AARCH64}

  ${SYS_INC_DIR}/gstreamer-1.0
  ${SYS_INC_DIR}/glib-2.0
  ${SYS_LIB_DIR}/glib-2.0/include

  ${JETSON_MMAPI_INC_DIR}

  ${VPI_INC_DIR}
  ${CUDA_INC_DIR}
)

# OpenCV headers (sysroot)
set(OPENCV4_INCLUDE_DIR ${CMAKE_SYSROOT}/usr/include/opencv4)
set(OPENCV2_INCLUDE_DIR ${CMAKE_SYSROOT}/usr/include)

if(EXISTS "${OPENCV4_INCLUDE_DIR}/opencv2/opencv.hpp")
  message(STATUS "Found OpenCV headers in: ${OPENCV4_INCLUDE_DIR}")
  target_include_directories(vpi_dense_of_nvbuffer PRIVATE ${OPENCV4_INCLUDE_DIR})
elseif(EXISTS "${OPENCV2_INCLUDE_DIR}/opencv2/opencv.hpp")
  message(STATUS "Found OpenCV headers in: ${OPENCV2_INCLUDE_DIR}")
  target_include_directories(vpi_dense_of_nvbuffer PRIVATE ${OPENCV2_INCLUDE_DIR})
else()
  message(FATAL_ERROR "OpenCV headers not found in sysroot.")
endif()

# Basic sanity checks
if(NOT EXISTS "${VPI_INC_DIR}/vpi/VPI.h")
  message(FATAL_ERROR "VPI headers not found: ${VPI_INC_DIR}/vpi/VPI.h")
endif()
if(NOT EXISTS "${CUDA_INC_DIR}/cuda_runtime.h")
  message(FATAL_ERROR "CUDA headers not found: ${CUDA_INC_DIR}/cuda_runtime.h")
endif()
if(NOT EXISTS "${CUPVA_LIB_DIR}")
  message(FATAL_ERROR "CUPVA lib dir not found: ${CUPVA_LIB_DIR}")
endif()

# -------------------------
# Find core libs
# -------------------------
find_library(GST_LIB      NAMES gstreamer-1.0 HINTS ${SYS_LIB_DIR} REQUIRED)
find_library(GST_BASE_LIB NAMES gstbase-1.0   HINTS ${SYS_LIB_DIR} REQUIRED)
find_library(GST_APP_LIB  NAMES gstapp-1.0    HINTS ${SYS_LIB_DIR} REQUIRED)
find_library(GLIB_LIB     NAMES glib-2.0      HINTS ${SYS_LIB_DIR} REQUIRED)
find_library(GOBJECT_LIB  NAMES gobject-2.0   HINTS ${SYS_LIB_DIR} REQUIRED)

find_library(NVBUFSURFACE_LIB NAMES nvbufsurface HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR} REQUIRED)

# -------------------------
# VPI / CUPVA / CUDA
# -------------------------
find_library(NVVPI_LIB NAMES nvvpi HINTS ${VPI_LIB_DIR} REQUIRED)
message(STATUS "Using NVVPI: ${NVVPI_LIB}")

find_library(CUDART_LIB NAMES cudart HINTS ${CUDA_LIB_DIR} REQUIRED)
message(STATUS "Using CUDART: ${CUDART_LIB}")

find_library(CUPVA_HOST_LIB
  NAMES cupva_host libcupva_host.so libcupva_host.so.2.3 libcupva_host.so.2.3.0
  HINTS ${CUPVA_LIB_DIR} ${VPI_PRIV_LIB_DIR}
  REQUIRED
)
message(STATUS "Using CUPVA_HOST: ${CUPVA_HOST_LIB}")

find_library(CUPVA_HOST_UTILS_LIB
  NAMES cupva_host_utils libcupva_host_utils.so libcupva_host_utils.so.2.3 libcupva_host_utils.so.2.3.0
  HINTS ${CUPVA_LIB_DIR} ${VPI_PRIV_LIB_DIR}
  REQUIRED
)
message(STATUS "Using CUPVA_HOST_UTILS: ${CUPVA_HOST_UTILS_LIB}")

# Explicit PVA deps (from Jetson ldd of libcupva_host.so)
find_library(NVPVA_LIB      NAMES nvpva      HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR} REQUIRED)
find_library(NVPVAINTF_LIB  NAMES nvpvaintf  HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR} REQUIRED)
find_library(NVPVAUMD_LIB   NAMES nvpvaumd   HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR} REQUIRED)

message(STATUS "Using NVPVA: ${NVPVA_LIB}")
message(STATUS "Using NVPVAINTF: ${NVPVAINTF_LIB}")
message(STATUS "Using NVPVAUMD: ${NVPVAUMD_LIB}")

# Optional NvSci deps (helpful for cross linkers)
find_library(NVSCIBUF_LIB    NAMES nvscibuf     HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR})
find_library(NVSCISYNC_LIB   NAMES nvscisync    HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR})
find_library(NVSCICOMMON_LIB NAMES nvscicommon  HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR})
find_library(NVSCIIPC_LIB    NAMES nvsciipc     HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR})

# EGL (VPI often needs it)
find_library(EGL_LIB NAMES EGL HINTS ${SYS_LIB_DIR} ${TEGRA_EGL_DIR} ${TEGRA_LIB_DIR} REQUIRED)
find_library(EGL_NVIDIA_LIB NAMES EGL_nvidia libEGL_nvidia.so.0 HINTS ${TEGRA_EGL_DIR} ${TEGRA_LIB_DIR})

# -------------------------
# OpenCV libs (sysroot only)
# -------------------------
set(_OPENCV_HINT_DIRS
  ${SYS_LIB_DIR}
  ${CMAKE_SYSROOT}/usr/lib
  ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu
)

macro(find_opencv_lib outvar)
  find_library(${outvar} NAMES ${ARGN} HINTS ${_OPENCV_HINT_DIRS})
endmacro()

find_opencv_lib(OPENCV_CORE_LIB      opencv_core opencv_core4)
find_opencv_lib(OPENCV_IMGPROC_LIB   opencv_imgproc opencv_imgproc4)
find_opencv_lib(OPENCV_IMGCODECS_LIB opencv_imgcodecs opencv_imgcodecs4)
find_opencv_lib(OPENCV_HIGHGUI_LIB   opencv_highgui opencv_highgui4)
find_opencv_lib(OPENCV_VIDEOIO_LIB   opencv_videoio opencv_videoio4)

foreach(L OPENCV_CORE_LIB OPENCV_IMGPROC_LIB OPENCV_IMGCODECS_LIB OPENCV_HIGHGUI_LIB)
  if(NOT ${L})
    message(FATAL_ERROR "Missing OpenCV lib: ${L}")
  endif()
endforeach()

set(OPENCV_LIBS
  ${OPENCV_CORE_LIB}
  ${OPENCV_IMGPROC_LIB}
  ${OPENCV_IMGCODECS_LIB}
  ${OPENCV_HIGHGUI_LIB}
)
if(OPENCV_VIDEOIO_LIB)
  list(APPEND OPENCV_LIBS ${OPENCV_VIDEOIO_LIB})
endif()

# -------------------------
# Link libraries
# NOTE: order matters for GNU ld in cross builds.
# -------------------------
target_link_libraries(vpi_dense_of_nvbuffer PRIVATE
  ${GST_LIB} ${GST_BASE_LIB} ${GST_APP_LIB}
  ${GLIB_LIB} ${GOBJECT_LIB}
  ${NVBUFSURFACE_LIB}

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

  ${CUDART_LIB}

  ${OPENCV_LIBS}

  pthread dl m
)

# -------------------------
# Linker options (cross build)
# -------------------------
target_link_options(vpi_dense_of_nvbuffer PRIVATE
  -Wl,--no-as-needed
  -Wl,--copy-dt-needed-entries

  # CRITICAL in your case:
  # CUPVA host keeps some PVA symbols undefined (resolved at runtime on target),
  # but the cross-linker wants them resolved at link-time.
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

  -Wl,-rpath,/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:/opt/nvidia/vpi2/lib/aarch64-linux-gnu:/opt/nvidia/vpi2/lib/aarch64-linux-gnu/priv:/opt/nvidia/cupva-2.3/lib/aarch64-linux-gnu:/usr/local/cuda/lib64
)
