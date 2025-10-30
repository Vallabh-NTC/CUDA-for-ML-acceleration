cmake_minimum_required(VERSION 3.18)

project(photo_edit_pipeline LANGUAGES CXX CUDA)

# --- CUDA toolkit root (same style as your working project)
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda CACHE PATH "CUDA Toolkit root")
endif()

# Jetson Orin (Ampere)
set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA architectures")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use shared CUDA runtime (prevents accidental static cudart linkage)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# ---- Sources ----
add_executable(photo_edit
    src/photo_edit.cu
)

set_source_files_properties(src/photo_edit.cu PROPERTIES LANGUAGE CUDA)

# ---- Includes ----
target_include_directories(photo_edit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb

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

target_link_libraries(photo_edit
    PRIVATE ${CUDART_LIBRARY}
)

# ---- Linker options (cross-link against Jetson sysroot) ----
target_link_options(photo_edit PRIVATE
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

# ---- Runtime RPATH on the target (optional) ----
set_target_properties(photo_edit PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "/usr/local/cuda/targets/aarch64-linux/lib:/usr/local/cuda/lib64"
)

# ---- Optional: harden ELF after build (if patchelf is available) ----
find_program(PATCHELF patchelf)
if (PATCHELF)
  add_custom_command(TARGET photo_edit POST_BUILD
    COMMAND ${PATCHELF} --set-interpreter /lib/aarch64-linux-gnu/ld-linux-aarch64.so.1 $<TARGET_FILE:photo_edit>
    COMMAND ${PATCHELF} --set-rpath "/usr/local/cuda/targets/aarch64-linux/lib:/usr/local/cuda/lib64" $<TARGET_FILE:photo_edit>
    VERBATIM
  )
endif()

# ---- Install (optional) ----
install(TARGETS photo_edit
    RUNTIME DESTINATION usr/local/bin
)

# ---- Diagnostics ----
message(STATUS "CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CMAKE_SYSROOT          = ${CMAKE_SYSROOT}")
message(STATUS "Found cudart:          ${CUDART_LIBRARY}")
