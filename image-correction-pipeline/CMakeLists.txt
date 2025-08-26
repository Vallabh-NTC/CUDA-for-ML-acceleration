# Nessun project() qui: eredita dal root

# ---- Libreria CUDA con i kernel ----
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
)

target_link_libraries(image_correction_kernels
  PUBLIC
    CUDA::cudart
            
)

# ---- Libreria C++ con la pipeline GStreamer ----
add_library(gstreamer_frontend STATIC
  src/gstreamer_pipeline.cpp
)

target_include_directories(gstreamer_frontend
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gstreamer_frontend
  PUBLIC
    PkgConfig::GST
    CUDA::cudart
    CUDA::toolkit
)
