cmake_minimum_required(VERSION 3.18)
project(vpi_of_nvivafilter LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# You already pass this in your command, keep it here as default
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

# ---- Sysroot must exist (toolchain should set it) ----
if(NOT CMAKE_SYSROOT)
  message(FATAL_ERROR "CMAKE_SYSROOT is not set. Use your toolchain or -DCMAKE_SYSROOT=/l4t/targetfs")
endif()

# ---- Paths ----
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

# ---- Target ----
add_library(vpi_of SHARED
  src/nvivafilter_vpi_of.cpp
  src/overlay.cu
  src/egl_copy.cu
  src/mv_reduce.cu
  
)
set_target_properties(vpi_of PROPERTIES OUTPUT_NAME "vpi_of")

# Ensure .cu is compiled as CUDA
set_source_files_properties(src/overlay.cu   PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/egl_copy.cu  PROPERTIES LANGUAGE CUDA)

# ---- Includes ----
target_include_directories(vpi_of PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include

  ${SYS_INC_DIR}
  ${SYS_INC_DIR_AARCH64}

  ${SYS_INC_DIR}/gstreamer-1.0
  ${SYS_INC_DIR}/glib-2.0
  ${SYS_LIB_DIR}/glib-2.0/include

  ${JETSON_MMAPI_INC_DIR}

  ${VPI_INC_DIR}
  ${CUDA_INC_DIR}
)

# ---- Libraries ----
find_library(GST_LIB            NAMES gstreamer-1.0     HINTS ${SYS_LIB_DIR} REQUIRED)
find_library(GST_BASE_LIB       NAMES gstbase-1.0       HINTS ${SYS_LIB_DIR} REQUIRED)
find_library(GST_APP_LIB        NAMES gstapp-1.0        HINTS ${SYS_LIB_DIR} REQUIRED)
find_library(GST_ALLOCATORS_LIB NAMES gstallocators-1.0 HINTS ${SYS_LIB_DIR} REQUIRED)
find_library(GLIB_LIB           NAMES glib-2.0          HINTS ${SYS_LIB_DIR} REQUIRED)
find_library(GOBJECT_LIB        NAMES gobject-2.0       HINTS ${SYS_LIB_DIR} REQUIRED)

find_library(NVVPI_LIB          NAMES nvvpi            HINTS ${VPI_LIB_DIR} REQUIRED)

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

find_library(NVPVA_LIB      NAMES nvpva      HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR} REQUIRED)
find_library(NVPVAINTF_LIB  NAMES nvpvaintf  HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR} REQUIRED)
find_library(NVPVAUMD_LIB   NAMES nvpvaumd   HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR} REQUIRED)

find_library(NVSCIBUF_LIB    NAMES nvscibuf     HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR})
find_library(NVSCISYNC_LIB   NAMES nvscisync    HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR})
find_library(NVSCICOMMON_LIB NAMES nvscicommon  HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR})
find_library(NVSCIIPC_LIB    NAMES nvsciipc     HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR})

find_library(EGL_LIB        NAMES EGL        HINTS ${SYS_LIB_DIR} ${TEGRA_EGL_DIR} ${TEGRA_LIB_DIR} REQUIRED)
find_library(EGL_NVIDIA_LIB NAMES EGL_nvidia libEGL_nvidia.so.0 HINTS ${TEGRA_EGL_DIR} ${TEGRA_LIB_DIR})

find_library(CUDA_LIB       NAMES cuda       HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR} REQUIRED)
find_library(CUDART_LIB     NAMES cudart     HINTS ${CUDA_LIB_DIR})

# ---- Compile options ----
target_compile_options(vpi_of PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# ---- Link ----
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

# ---- Linker options / rpath ----
target_link_options(vpi_of PRIVATE
  -Wl,--no-as-needed
  -Wl,--copy-dt-needed-entries
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

install(TARGETS vpi_of LIBRARY DESTINATION usr/local/lib/nvivafilter)
