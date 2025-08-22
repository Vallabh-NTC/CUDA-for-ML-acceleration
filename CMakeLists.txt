cmake_minimum_required(VERSION 3.18)  # Modern CMake for CUDA
project(CNN_Project LANGUAGES CXX CUDA)

# Enable CUDA and C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Option for debug build
option(BUILD_DEBUG "Enable debug build" ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/main.cu
    src/utils.cu
)

# Create executable
add_executable(cnn_app ${SOURCES})

# Set CUDA architectures (adjust to your GPU)
set_target_properties(cnn_app PROPERTIES
    CUDA_ARCHITECTURES "75"   # e.g., Jetson Orin: Ampere/Tensor cores; adjust if needed
)

# Compiler flags
if(BUILD_DEBUG)
    target_compile_options(cnn_app PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O0 -g>
        $<$<COMPILE_LANGUAGE:CUDA>:-G -lineinfo -Xcompiler=-Wall,-Wextra,-O0>
    )
endif()
