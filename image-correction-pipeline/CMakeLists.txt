cmake_minimum_required(VERSION 3.18)
project(camera_rectify LANGUAGES CXX CUDA)

# --- Global settings ---
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 72 CACHE STRING "SM arch for CUDA") # Xavier/Xavier NX

# Allow a sysroot for cross builds (e.g., /l4t/targetfs)
set(SYSROOT "$ENV{SYSROOT}" CACHE PATH "Jetson target rootfs (sysroot) for cross-compiling")

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
set(L4T_LIB_DIRS)
if(SYSROOT)
  list(APPEND L4T_LIB_DIRS
    ${SYSROOT}/usr/lib/aarch64-linux-gnu
    ${SYSROOT}/lib/aarch64-linux-gnu
    ${SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
    ${SYSROOT}/usr/lib/tegra
  )
else()
  list(APPEND L4T_LIB_DIRS
    /usr/lib/aarch64-linux-gnu
    /lib/aarch64-linux-gnu
    /usr/lib/aarch64-linux-gnu/tegra
    /usr/lib/tegra
  )
endif()

# Jetson Multimedia API include dir (sysroot-aware)
_sysroot_hint(JETSON_MMAPI_INC "/usr/src/jetson_multimedia_api/include")

# ---------- CUDA Toolkit / cudart ----------
set(_cuda_ok FALSE)
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND AND TARGET CUDA::cudart)
  set(_cuda_ok TRUE)
else()
  set(_cudart_candidates)
  if(SYSROOT)
    list(APPEND _cudart_candidates
      "${SYSROOT}/usr/local/cuda-11.4/targets/aarch64-linux/lib/libcudart.so"
      "${SYSROOT}/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so")
  else()
    list(APPEND _cudart_candidates
      "/usr/local/cuda-11.4/targets/aarch64-linux/lib/libcudart.so"
      "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so")
  endif()
  foreach(p IN LISTS _cudart_candidates)
    if(EXISTS "${p}")
      add_library(CUDA::cudart SHARED IMPORTED)
      set_target_properties(CUDA::cudart PROPERTIES IMPORTED_LOCATION "${p}")
      if(SYSROOT)
        set_property(TARGET CUDA::cudart APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
          "${SYSROOT}/usr/local/cuda-11.4/targets/aarch64-linux/include"
          "${SYSROOT}/usr/local/cuda/targets/aarch64-linux/include")
      else()
        set_property(TARGET CUDA::cudart APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
          "/usr/local/cuda-11.4/targets/aarch64-linux/include"
          "/usr/local/cuda/targets/aarch64-linux/include")
      endif()
      set(_cuda_ok TRUE)
      break()
    endif()
  endforeach()
endif()

if(NOT _cuda_ok)
  message(FATAL_ERROR "Could not locate CUDA toolkit or aarch64 libcudart.so (set SYSROOT or install CUDA).")
endif()

# ---------- EGL (host stub) ----------
# EGL headers come dal sysroot
if(SYSROOT)
  include_directories(${SYSROOT}/usr/include ${SYSROOT}/usr/include/aarch64-linux-gnu)
else()
  include_directories(/usr/include /usr/include/aarch64-linux-gnu)
endif()

# EGL lib dal container host (Mesa stub, x86-64)
unset(EGL_LIB CACHE)
find_library(EGL_LIB EGL HINTS /usr/lib/x86_64-linux-gnu REQUIRED)

# ---------- CUDA Driver stub (host x86-64) ----------
unset(CUDA_DRIVER_LIB CACHE)
find_library(CUDA_DRIVER_LIB cuda
  HINTS /usr/local/cuda-11.4/targets/x86_64-linux/lib/stubs
  REQUIRED)
message(STATUS "Using CUDA driver stub: ${CUDA_DRIVER_LIB}")

# ---------- CUDA kernels ----------
add_library(image_correction_kernels STATIC
  src/image_correction.cu
)
target_include_directories(image_correction_kernels PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include
)
set_target_properties(image_correction_kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(image_correction_kernels PUBLIC
  CUDA::cudart
)

# ---------- nvivafilter plugin (.so) ----------
add_library(rectify_nvivafilter SHARED
  src/nvivafilter_rectify.cu
)

target_include_directories(rectify_nvivafilter PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include
  ${JETSON_MMAPI_INC}
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
  ${CUDA_DRIVER_LIB}   # x86-64 stub
  ${EGL_LIB}           # x86-64 stub
)

# NB: nvbufsurface solo headers, non linkato in cross build
# (si risolve a runtime sul Jetson)

install(TARGETS rectify_nvivafilter
  LIBRARY DESTINATION /opt/lib
)
