cmake_minimum_required(VERSION 3.18)

project(udp_recorder_pipeline LANGUAGES CXX CUDA)

# --- CUDA toolkit root (same style as your working project)
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda CACHE PATH "CUDA Toolkit root")
endif()

# Jetson Orin (Ampere)
set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use shared CUDA runtime (prevents accidental static cudart linkage)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# ---- Sources ----
add_executable(udp_decoder
    src/BitExtractor.cpp
    src/LWI01.cpp
    src/UdpReceiver.cpp
    src/main.cpp
    src/KernelExec.cu
    src/UdpSender.cpp
    src/Motor20.cpp
    src/BrakeEV01.cpp
    src/ESP21.cpp
    src/SARA_10.cpp
    src/SARA_08.cpp
)

set_source_files_properties(src/KernelExec.cu PROPERTIES LANGUAGE CUDA)

# ---- Includes ----
target_include_directories(udp_decoder PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src

    # Sysroot headers (match target)
    ${CMAKE_SYSROOT}/usr/include
    ${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu
    /l4t/targetfs/usr/include
    /l4t/targetfs/usr/include/aarch64-linux-gnu

    # CUDA headers
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/include
)

# ---- Libraries ----
# CUDA runtime (shared)
find_library(CUDART_LIBRARY cudart
    HINTS
        ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CMAKE_SYSROOT}/usr/local/cuda/lib64
)
if(NOT CUDART_LIBRARY)
    message(FATAL_ERROR "Could not find libcudart.so — check CUDA_TOOLKIT_ROOT_DIR and CMAKE_SYSROOT.")
endif()

# -------------------- GStreamer (gst + appsink) --------------------
find_package(PkgConfig REQUIRED)

# If you are cross-compiling and pkg-config must point to the sysroot,
# ensure PKG_CONFIG_SYSROOT_DIR / PKG_CONFIG_LIBDIR are set in your toolchain/env.
pkg_check_modules(GST REQUIRED
    gstreamer-1.0
    gstreamer-app-1.0
)

# Add include dirs and compile flags for GStreamer
target_include_directories(udp_decoder PRIVATE
    ${GST_INCLUDE_DIRS}
)

target_compile_options(udp_decoder PRIVATE
    ${GST_CFLAGS_OTHER}
)

# Add link dirs (if any) + libs
target_link_directories(udp_decoder PRIVATE
    ${GST_LIBRARY_DIRS}
)

target_link_libraries(udp_decoder
    PRIVATE
        ${CUDART_LIBRARY}
        ${GST_LIBRARIES}
)

# ---- Linker options (cross-link against Jetson sysroot) ----
target_link_options(udp_decoder PRIVATE
    # Use the target sysroot
    "SHELL:--sysroot=${CMAKE_SYSROOT}"

    # Make sure glibc and friends are found in the sysroot
    "-Wl,-rpath-link,${CMAKE_SYSROOT}/lib/aarch64-linux-gnu"
    "-Wl,-rpath-link,${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu"
    "-L${CMAKE_SYSROOT}/lib/aarch64-linux-gnu"
    "-L${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu"

    # ✅ Use the real on-device dynamic loader (avoid embedding the sysroot path)
    "-Wl,--dynamic-linker=/lib/aarch64-linux-gnu/ld-linux-aarch64.so.1"

    # CUDA device libs (for cudart)
    -L${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib
)

# ---- Install (optional) ----
install(TARGETS udp_decoder
    RUNTIME DESTINATION usr/local/bin
)

# ---- Diagnostics ----
message(STATUS "CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CMAKE_SYSROOT          = ${CMAKE_SYSROOT}")
message(STATUS "Found cudart:          ${CUDART_LIBRARY}")
message(STATUS "GStreamer includes:    ${GST_INCLUDE_DIRS}")
message(STATUS "GStreamer libs:        ${GST_LIBRARIES}")
message(STATUS "GStreamer lib dirs:    ${GST_LIBRARY_DIRS}")
