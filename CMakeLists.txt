cmake_minimum_required(VERSION 3.18)
project(JetsonMonorepo LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Xavier (Volta) SM_72 by default; override with -DCUDA_ARCH=87 for Orin
set(CUDA_ARCH "72" CACHE STRING "CUDA SM arch (72=Xavier, 87=Orin)")

# Which projects to build
option(BUILD_IMAGE_PIPELINE "Build image-pipeline-project" ON)
option(BUILD_HAND_GESTURE   "Build hand-gesture-project"   OFF)

# Core library is always available
add_subdirectory(cuda-framework)

# Selected projects
if(BUILD_IMAGE_PIPELINE)
  add_subdirectory(image-pipeline-project)
endif()

if(BUILD_HAND_GESTURE)
  add_subdirectory(hand-gesture-project)
endif()

message(STATUS "CUDA_ARCH=${CUDA_ARCH}")
message(STATUS "BUILD_IMAGE_PIPELINE=${BUILD_IMAGE_PIPELINE}")
message(STATUS "BUILD_HAND_GESTURE=${BUILD_HAND_GESTURE}")
