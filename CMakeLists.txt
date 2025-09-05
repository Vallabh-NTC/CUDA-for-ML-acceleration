cmake_minimum_required(VERSION 3.18)
project(ImageCorrectionPipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)

# GStreamer come target IMPORTED (propaga include + libs)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET
  gstreamer-1.0
  gstreamer-app-1.0
  gstreamer-video-1.0
)

# Sotto-progetto con le librerie
add_subdirectory(image-correction-pipeline)

# ---- Executable principale (CUDA) ----
add_executable(main_app
  src/main.cu
)

target_include_directories(main_app
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Linka TUTTO ciò che serve a main.cu:
target_link_libraries(main_app
  PRIVATE
    PkgConfig::GST            # <gst/gst.h> + libs
    gstreamer_frontend        # implementazione della pipeline (cpp)
    image_correction_kernels  # i tuoi kernel CUDA
    CUDA::cudart
)

# Opzionale: se serve separable compilation
set_target_properties(main_app PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
