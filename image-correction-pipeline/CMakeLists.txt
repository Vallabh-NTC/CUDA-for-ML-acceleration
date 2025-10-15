# =============================================================================
# CMake build script for the "image-correction-pipeline" project
#
# Builds a GStreamer "nvivafilter" plugin with CUDA support
# for NVIDIA Jetson (cross-compiled from x86 host).
#
# Produces:
#   libnvivafilter_imagecorrection.so
# Install path:
#   /usr/lib/aarch64-linux-gnu/gstreamer-1.0/
#
# Sources:
#   - kernel_rectify.cu           : fisheye undistortion
#   - wire_lineremoval.cu         : optional wire inpaint
#   - color_ops.cu                : tone/color
#   - nvivafilter_imagecorrection.cpp : plugin integration
#   - runtime_controls.cpp        : JSON config hot-reload
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(image_correction_pipeline LANGUAGES CXX CUDA)

# ----------------------------------------------------------------------------- 
# CUDA toolkit root
# -----------------------------------------------------------------------------
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda CACHE PATH "CUDA Toolkit root")
endif()

# GPU arch: respect caller; default to 87 (Orin) if not set
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA architectures")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ----------------------------------------------------------------------------- 
# Sources
# -----------------------------------------------------------------------------
set(CUDA_SOURCES
    src/kernel_rectify.cu
    src/wire_lineremoval.cu
    src/color_ops.cu
)

set(CPP_SOURCES
    src/nvivafilter_imagecorrection.cpp
    src/runtime_controls.cpp
)

add_library(nvivafilter_imagecorrection SHARED
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

# ----------------------------------------------------------------------------- 
# Profiling-friendly flags (NVTX/range visibility)
# -----------------------------------------------------------------------------
target_compile_options(nvivafilter_imagecorrection PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-fno-omit-frame-pointer>
)
target_compile_definitions(nvivafilter_imagecorrection PRIVATE NVTX_ENABLE=1)

# ----------------------------------------------------------------------------- 
# Include directories
# -----------------------------------------------------------------------------
target_include_directories(nvivafilter_imagecorrection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /l4t/targetfs/usr/src/jetson_multimedia_api/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/include
    ${CMAKE_SYSROOT}/usr/include
    ${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu
)

# ----------------------------------------------------------------------------- 
# Libraries
# -----------------------------------------------------------------------------
find_library(CUDART_LIBRARY cudart
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
          ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib
          ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib/stubs
          ${CMAKE_SYSROOT}/usr/local/cuda/lib64
)
find_library(CUDA_DRIVER_LIBRARY cuda
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
          ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib
          ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib/stubs
          ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
)

# NVTX (nvToolsExt) — used by nvtxRangePush/Pop
find_library(NVTOOLSEXT_LIBRARY nvToolsExt
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
          ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib
          ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib/stubs
          ${CMAKE_SYSROOT}/usr/local/cuda/lib64
          ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu
)

find_library(LIB_EGL EGL HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_X11 X11 HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_XEXT Xext HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_BSD bsd HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_RT rt HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_PTHREAD pthread HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_M m HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
find_library(LIB_DL dl HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)

target_link_libraries(nvivafilter_imagecorrection
    ${CUDART_LIBRARY}
    ${CUDA_DRIVER_LIBRARY}
    ${LIB_X11}
    ${LIB_XEXT}
    ${LIB_BSD}
    ${LIB_RT}
    ${LIB_PTHREAD}
    ${LIB_M}
    ${LIB_DL}
)

# Link NVTX if found (for your NVTX v3 ranges)
if(NVTOOLSEXT_LIBRARY)
    target_link_libraries(nvivafilter_imagecorrection PRIVATE ${NVTOOLSEXT_LIBRARY})
else()
    message(WARNING "nvToolsExt (NVTX) not found — NVTX ranges will not link.")
endif()

# ----------------------------------------------------------------------------- 
# Sysroot library paths
# -----------------------------------------------------------------------------
target_link_options(nvivafilter_imagecorrection PRIVATE
    -L${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu
    -L${CMAKE_SYSROOT}/lib/aarch64-linux-gnu
    -L${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib
    -L${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib/stubs
)

# ----------------------------------------------------------------------------- 
# Install
# -----------------------------------------------------------------------------
install(TARGETS nvivafilter_imagecorrection
    LIBRARY DESTINATION lib/aarch64-linux-gnu/gstreamer-1.0/
)

# ========================= Deploy folder & run scripts ========================
# Creates: build/.../deploy/{libnvivafilter_imagecorrection.so, run.sh, run_profile.sh)
set(DEPLOY_DIR "${CMAKE_CURRENT_BINARY_DIR}/deploy")
file(MAKE_DIRECTORY "${DEPLOY_DIR}")

# 1) Copy the .so into deploy/
add_custom_command(
    TARGET nvivafilter_imagecorrection POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:nvivafilter_imagecorrection>"
            "${DEPLOY_DIR}/libnvivafilter_imagecorrection.so"
    COMMENT "Copy plugin to deploy/"
)

# 2) Create run.sh (plain gst-launch, minimal env; allows env overrides)
set(RUN_SH "${DEPLOY_DIR}/run.sh")
file(WRITE  "${RUN_SH}" "#!/bin/bash\n")
file(APPEND "${RUN_SH}" "set -e\n")
file(APPEND "${RUN_SH}" "DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n")
file(APPEND "${RUN_SH}" "SO=\"$DIR/libnvivafilter_imagecorrection.so\"\n")
file(APPEND "${RUN_SH}" ": \${HOST_IP:=192.168.10.201}\n")
file(APPEND "${RUN_SH}" ": \${HOST_PORT:=5000}\n")
file(APPEND "${RUN_SH}" ": \${WIDTH:=1920}\n")
file(APPEND "${RUN_SH}" ": \${HEIGHT:=1080}\n")
file(APPEND "${RUN_SH}" ": \${FPS:=30}\n")
file(APPEND "${RUN_SH}" ": \${BITRATE:=12000000}\n")
file(APPEND "${RUN_SH}" ": \${SENSOR_ID:=0}\n")
file(APPEND "${RUN_SH}" "echo \"→ Using SO: $SO\" >&2\n")
file(APPEND "${RUN_SH}" "echo \"→ Camera sensor-id=\${SENSOR_ID} → \${HOST_IP}:\${HOST_PORT} (\${WIDTH}x\${HEIGHT}@\${FPS})\" >&2\n")
file(APPEND "${RUN_SH}" "exec gst-launch-1.0 -e -v \\\n")
file(APPEND "${RUN_SH}" "  nvarguscamerasrc sensor-id=\${SENSOR_ID} ! \\\n")
file(APPEND "${RUN_SH}" "  \"video/x-raw(memory:NVMM),format=NV12,width=\${WIDTH},height=\${HEIGHT},framerate=\${FPS}/1\" ! \\\n")
file(APPEND "${RUN_SH}" "  nvivafilter customer-lib-name=$SO pre-process=false cuda-process=true post-process=false ! \\\n")
file(APPEND "${RUN_SH}" "  \"video/x-raw(memory:NVMM),format=NV12\" ! \\\n")
file(APPEND "${RUN_SH}" "  nvv4l2h264enc bitrate=\${BITRATE} insert-sps-pps=true idrinterval=30 preset-level=4 ! \\\n")
file(APPEND "${RUN_SH}" "  h264parse ! rtph264pay pt=96 config-interval=1 ! \\\n")
file(APPEND "${RUN_SH}" "  udpsink host=\${HOST_IP} port=\${HOST_PORT} sync=false async=false \"\$@\"\n")
execute_process(COMMAND ${CMAKE_COMMAND} -E chmod +x "${RUN_SH}")

# 3) Create run_profile.sh (nsys wrapper; honors cudaProfilerStart/Stop)
set(RUN_PROF_SH "${DEPLOY_DIR}/run_profile.sh")
file(WRITE  "${RUN_PROF_SH}" "#!/bin/bash\n")
file(APPEND "${RUN_PROF_SH}" "set -e\n")
file(APPEND "${RUN_PROF_SH}" "DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n")
file(APPEND "${RUN_PROF_SH}" "NSYS=$(command -v nsys || true)\n")
file(APPEND "${RUN_PROF_SH}" "if [ -z \"$NSYS\" ]; then echo \"❌ nsys not found on this device\" >&2; exit 1; fi\n")
file(APPEND "${RUN_PROF_SH}" "REP=\"$DIR/icp_$(date +%F_%H-%M)\"\n")
file(APPEND "${RUN_PROF_SH}" "echo \"→ Profiling to: ${REP}.nsys-rep\" >&2\n")
file(APPEND "${RUN_PROF_SH}" "exec \"$NSYS\" profile \\\n")
file(APPEND "${RUN_PROF_SH}" "  --output \"$REP\" \\\n")
file(APPEND "${RUN_PROF_SH}" "  --trace=cuda,nvtx,osrt \\\n")
file(APPEND "${RUN_PROF_SH}" "  --capture-range=cudaProfilerApi \\\n")
file(APPEND "${RUN_PROF_SH}" "  --stop-on-exit=true \\\n")
file(APPEND "${RUN_PROF_SH}" "  \"$DIR/run.sh\" \"$@\"\n")
execute_process(COMMAND ${CMAKE_COMMAND} -E chmod +x "${RUN_PROF_SH}")

# 4) Convenience target to allow: cmake --build . --target deploy
add_custom_target(deploy ALL
    DEPENDS nvivafilter_imagecorrection
    COMMENT "Deploy folder ready at: ${DEPLOY_DIR}"
)
