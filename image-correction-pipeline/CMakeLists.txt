cmake_minimum_required(VERSION 3.18)
project(ImagePipeline LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# Always build the CUDA part (kernel library or object)
add_library(image_filter STATIC src/image_filter.cu)
target_include_directories(image_filter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(image_filter PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "${CUDA_ARCH}"
)
target_link_libraries(image_filter PUBLIC cuda_framework CUDA::cudart)

# If a GStreamer front-end exists, build an app that uses it
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/camera_pipeline.cpp")
  # GStreamer (only for the app)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)

  add_executable(image_pipeline_app
    src/camera_pipeline.cpp
  )
  target_include_directories(image_pipeline_app PRIVATE ${GST_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(image_pipeline_app PRIVATE ${GST_CFLAGS_OTHER})
  target_link_libraries(image_pipeline_app PRIVATE ${GST_LIBRARIES} image_filter)

  message(STATUS "Building image_pipeline_app with GStreamer.")
else()
  message(WARNING "camera_pipeline.cpp not found; building only image_filter library.")
endif()
