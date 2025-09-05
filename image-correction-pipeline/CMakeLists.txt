cmake_minimum_required(VERSION 3.18)
project(camera_rectify LANGUAGES CXX CUDA)

# --- Global settings ---
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 72 CACHE STRING "SM arch for CUDA") # Xavier/Xavier NX

# Detect SYSROOT from env or fallback to /l4t/targetfs
set(SYSROOT "$ENV{SYSROOT}" CACHE PATH "Jetson target rootfs (sysroot) for cross-compiling")
if(NOT SYSROOT)
  set(SYSROOT "/l4t/targetfs")
endif()

# ---------- Helpers ----------
function(_sysroot_hint out_var)
  set(abs_path "${ARGV1}")
  if(SYSROOT)
    set(${out_var} "${SYSROOT}${abs_path}" PARENT_SCOPE)
  else()
    set(${out_var} "${abs_path}" PARENT_SCOPE)
  endif()
endfunction()

# Candidate library dirs inside the target filesystem
set(L4T_LIB_DIRS
  ${SYSROOT}/usr/lib/aarch64-linux-gnu
  ${SYSROOT}/lib/aarch64-linux-gnu
  ${SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
  ${SYSROOT}/usr/lib/tegra
)

# Jetson Multimedia API include dir (sysroot-aware)
_sysroot_hint(JETSON_MMAPI_INC "/usr/src/jetson_multimedia_api/include")

# ---------- CUDA Toolkit / cudart (sysroot-only, no host libs) ----------
# We avoid find_package(CUDAToolkit) to prevent host contamination.
# Import cudart directly from the sysroot and expose include dirs.

# Locate sysroot CUDA root
set(_SYS_CUDA_CANDIDATES
  "${SYSROOT}/usr/local/cuda-11.4"
  "${SYSROOT}/usr/local/cuda"
)
set(_SYS_CUDA_FOUND FALSE)
foreach(_c IN LISTS _SYS_CUDA_CANDIDATES)
  if(EXISTS "${_c}/targets/aarch64-linux/lib/libcudart.so")
    set(JETSON_CUDA_ROOT "${_c}")
    set(_SYS_CUDA_FOUND TRUE)
    break()
  endif()
endforeach()

if(NOT _SYS_CUDA_FOUND)
  message(FATAL_ERROR "Could not find aarch64 CUDA in sysroot. Checked: ${_SYS_CUDA_CANDIDATES}")
endif()

# Import cudart from sysroot with an absolute path
add_library(CUDA::cudart SHARED IMPORTED)
set_target_properties(CUDA::cudart PROPERTIES
  IMPORTED_LOCATION "${JETSON_CUDA_ROOT}/targets/aarch64-linux/lib/libcudart.so"
  INTERFACE_INCLUDE_DIRECTORIES "${JETSON_CUDA_ROOT}/targets/aarch64-linux/include"
)

# ---------- EGL / Driver API (from Jetson sysroot) ----------
unset(EGL_LIB CACHE)
find_library(EGL_LIB EGL HINTS ${L4T_LIB_DIRS} REQUIRED)

unset(CUDA_DRIVER_LIB CACHE)
find_library(CUDA_DRIVER_LIB cuda HINTS ${L4T_LIB_DIRS} REQUIRED)

# ---------- CUDA kernels ----------
add_library(image_correction_kernels STATIC
  src/image_correction.cu
)
target_include_directories(image_correction_kernels PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include
  "${JETSON_CUDA_ROOT}/targets/aarch64-linux/include"
)
set_target_properties(image_correction_kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(image_correction_kernels PUBLIC CUDA::cudart)

# ---------- nvivafilter plugin (.so) ----------
add_library(rectify_nvivafilter SHARED
  src/nvivafilter_rectify.cu
)

target_include_directories(rectify_nvivafilter PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include
  ${JETSON_MMAPI_INC}
  "${JETSON_CUDA_ROOT}/targets/aarch64-linux/include"
)

set_target_properties(rectify_nvivafilter PROPERTIES
  OUTPUT_NAME rectify
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  CUDA_SEPARABLE_COMPILATION ON
  BUILD_RPATH "/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra"
  INSTALL_RPATH "/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra"
)

target_link_libraries(rectify_nvivafilter PRIVATE
  image_correction_kernels
  CUDA::cudart
  ${CUDA_DRIVER_LIB}
  ${EGL_LIB}
)

# nvbufsurface: headers only, resolves at runtime on Jetson
install(TARGETS rectify_nvivafilter
  LIBRARY DESTINATION /opt/lib
)


