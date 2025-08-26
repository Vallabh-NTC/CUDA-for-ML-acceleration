cmake_minimum_required(VERSION 3.18)
project(ImageCorrectionPipeline LANGUAGES CXX CUDA)

# Make sure the root created the target
if(NOT TARGET cuda_framework)
  message(FATAL_ERROR "Target cuda_framework not found. Root must add_subdirectory(cuda-framework) first.")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)

# ---- CUDA kernels (library) ----
add_library(image_correction_kernels STATIC
    src/image_correction.cu
)
target_include_directories(image_correction_kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
)
set_target_properties(image_correction_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CUDA_ARCH}"
)
# Link CMake TARGETS, not raw -l names
target_link_libraries(image_correction_kernels
    PUBLIC
        cuda_framework
        CUDA::cudart
)

# ---- GStreamer front-end app ----
add_executable(image_pipeline_app
    src/camera_pipeline.cpp
)
target_include_directories(image_pipeline_app
    PRIVATE
        ${GST_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
)
target_compile_options(image_pipeline_app PRIVATE ${GST_CFLAGS_OTHER})
target_link_libraries(image_pipeline_app
    PRIVATE
        ${GST_LIBRARIES}
        image_correction_kernels
)
