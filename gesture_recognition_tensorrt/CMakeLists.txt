cmake_minimum_required(VERSION 3.14)
project(trt_cam_engine CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# OpenCV
find_package(OpenCV REQUIRED)

# TensorRT libs
find_library(NVINFER_LIB nvinfer)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin)

# CUDA headers + runtime
find_package(CUDAToolkit QUIET)
if (CUDAToolkit_FOUND)
    message(STATUS "Using CUDAToolkit: ${CUDAToolkit_VERSION}")
    set(CUDART_LIB CUDA::cudart)
else()
    # Force Jetson paths
    set(CUDA_INCLUDE_DIR "/usr/local/cuda/include" CACHE PATH "CUDA include dir")
    include_directories(${CUDA_INCLUDE_DIR})
    find_library(CUDART_LIB cudart
        HINTS
            /usr/local/cuda/targets/aarch64-linux/lib
            /usr/local/cuda/lib64
            /usr/lib/aarch64-linux-gnu
            /usr/lib
    )
endif()

if (NOT NVINFER_LIB)
    message(FATAL_ERROR "libnvinfer.so not found")
endif()
if (NOT CUDART_LIB)
    message(FATAL_ERROR "libcudart.so not found (set -DCUDART_LIB=/full/path/libcudart.so)")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    /usr/include
    /usr/include/aarch64-linux-gnu
    # CUDA_INCLUDE_DIR is already added above if needed
)

add_executable(trt_cam_engine main_engine.cpp)

target_link_libraries(trt_cam_engine
    ${OpenCV_LIBS}
    ${NVINFER_LIB}
    ${NVINFER_PLUGIN_LIB}
    ${CUDART_LIB}
)

message(STATUS "OpenCV: ${OpenCV_LIBS}")
message(STATUS "nvinfer: ${NVINFER_LIB}")
message(STATUS "nvinfer_plugin: ${NVINFER_PLUGIN_LIB}")
message(STATUS "cudart: ${CUDART_LIB}")
if (DEFINED CUDA_INCLUDE_DIR)
    message(STATUS "CUDA include: ${CUDA_INCLUDE_DIR}")
endif()
