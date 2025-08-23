cmake_minimum_required(VERSION 3.18)
project(CNN_Project LANGUAGES CXX CUDA)

# Enable C++17 and CUDA17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

option(BUILD_DEBUG "Enable debug build" ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/include)           # cuDNN headers

# Source files
set(SOURCES
    src/main.cu
    src/utils.cu
)

# Create executable
add_executable(cnn_app ${SOURCES})

# CUDA architecture for Jetson Orin (Ampere)
set_target_properties(cnn_app PROPERTIES
    CUDA_ARCHITECTURES "87"
)

# Link to CUDA, cuBLAS, cuBLAS Lt, cuDNN
target_link_libraries(cnn_app
    /usr/local/cuda/lib64/libcudart.so
    /usr/local/cuda/lib64/libcublas.so
    /usr/local/cuda/lib64/libcublasLt.so
    /usr/lib/aarch64-linux-gnu/libcudnn.so
)

# Compiler flags for debug
if(BUILD_DEBUG)
    target_compile_options(cnn_app PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O0 -g>
        $<$<COMPILE_LANGUAGE:CUDA>:-G -lineinfo -Xcompiler=-Wall,-Wextra,-O0>
    )
endif()
