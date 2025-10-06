# =============================================================================
# Root CMakeLists.txt
#
# Dispatcher for sub-projects in this workspace.
# Example: CUDA-for-ML-acceleration can contain multiple CUDA/GStreamer modules.
#
# Usage:
#   mkdir build && cd build
#   cmake .. -DPROJECT=image-correction-pipeline \
#            -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/Toolchain_aarch64_l4t.cmake
# =============================================================================

cmake_minimum_required(VERSION 3.18)

# Root project; supports both host and CUDA languages.
# Actual cross-compilation settings are controlled via toolchain file.
project(CUDA_for_ML_acceleration LANGUAGES C CXX CUDA)

# Enforce modern C++/CUDA defaults at root level (can be overridden in subprojects).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# -----------------------------------------------------------------------------
# PROJECT selection mechanism
# -----------------------------------------------------------------------------
# The -DPROJECT=<subdir> variable selects which module to build.
# Example:
#   cmake .. -DPROJECT=image-correction-pipeline
set(PROJECT "" CACHE STRING "Subproject to build (e.g. image-correction-pipeline)")

# Require explicit selection
if (PROJECT STREQUAL "")
    message(FATAL_ERROR "Please specify -DPROJECT=<subdir>, e.g. -DPROJECT=image-correction-pipeline")
endif()

# Check subproject existence and include its CMakeLists.txt
if (EXISTS "${CMAKE_SOURCE_DIR}/${PROJECT}/CMakeLists.txt")
    add_subdirectory(${PROJECT})
else()
    message(FATAL_ERROR "Unknown project: ${PROJECT}")
endif()


