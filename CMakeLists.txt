cmake_minimum_required(VERSION 3.18)
project(CUDA_ML_Acceleration LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Xavier default; override with -DCUDA_ARCH=87 for Orin
set(CUDA_ARCH "72" CACHE STRING "CUDA SM arch (72=Xavier, 87=Orin)")

find_package(CUDAToolkit REQUIRED)  # gives CUDA::cudart, etc.

# -------- Core CUDA framework library (VISIBLE to subdirs) --------
# Headers live under include/cnn/, but we expose the parent include/
add_library(cuda_framework STATIC
    src/utils.cu
)
target_include_directories(cuda_framework
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)
set_target_properties(cuda_framework PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CUDA_ARCH}"
)
target_link_libraries(cuda_framework
    PUBLIC
        CUDA::cudart
)

# (Optional) small app that uses the core lib; remove if you don't want it
if(EXISTS "${CMAKE_SOURCE_DIR}/src/main.cu")
    add_executable(cnn_app src/main.cu)
    set_target_properties(cnn_app PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CUDA_ARCH}"
    )
    target_link_libraries(cnn_app PRIVATE cuda_framework)
endif()

# -------- Projects (toggle as needed) --------
option(BUILD_IMAGE_CORRECTION "Build image-correction-pipeline" ON)

if(BUILD_IMAGE_CORRECTION)
    add_subdirectory(image-correction-pipeline)
endif()

message(STATUS "CUDA_ARCH=${CUDA_ARCH}")
message(STATUS "BUILD_IMAGE_CORRECTION=${BUILD_IMAGE_CORRECTION}")
