cmake_minimum_required(VERSION 3.18)
project(camera_rectify LANGUAGES CXX CUDA)

# --- Global settings ---
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# (Recommended) set SM arch; pick your platform
# set(CMAKE_CUDA_ARCHITECTURES 72)  # Xavier
# set(CMAKE_CUDA_ARCHITECTURES 87)  # Orin

find_package(PkgConfig REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS EGL)

# ---- CUDA kernels (unchanged) ----
add_library(image_correction_kernels STATIC
  src/image_correction.cu
)

target_include_directories(image_correction_kernels PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(image_correction_kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(image_correction_kernels PUBLIC
  CUDA::cudart
)

# ---- nvivafilter plugin (.so loaded at runtime) ----
add_library(rectify_nvivafilter SHARED
  src/nvivafilter_rectify.cu
)

# Resolve special libs reliably
find_library(CUDA_EGL_LIB cudaEGL HINTS /usr/local/cuda/lib64)
find_library(NVBUFSURFACE_LIB nvbufsurface
  HINTS /usr/lib/aarch64-linux-gnu/tegra /usr/lib/aarch64-linux-gnu)

target_include_directories(rectify_nvivafilter PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include
  /usr/src/jetson_multimedia_api/include
  /usr/include/aarch64-linux-gnu
)

set_target_properties(rectify_nvivafilter PROPERTIES
  OUTPUT_NAME rectify
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  CUDA_SEPARABLE_COMPILATION ON
)

# (Optional) small perf tweak
# target_compile_options(rectify_nvivafilter PRIVATE
#   $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
# )

target_link_libraries(rectify_nvivafilter PRIVATE
  image_correction_kernels
  CUDA::cudart
  ${CUDA_EGL_LIB}
  OpenGL::EGL
  ${NVBUFSURFACE_LIB}
)

install(TARGETS rectify_nvivafilter
  LIBRARY DESTINATION /opt/lib
)

# ---- (old frontend targets kept commented) ----
# add_library(gstreamer_frontend STATIC
#   src/gstreamer_pipeline.cpp
#   src/rtp_sender.cpp
# )
# target_include_directories(gstreamer_frontend PUBLIC
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
#   ${CMAKE_SOURCE_DIR}/include
# )
# target_link_libraries(gstreamer_frontend PUBLIC
#   PkgConfig::GST
#   CUDA::cudart
#   CUDA::toolkit
# )
