# =============================================================================
# CMake build script for the "image-correction-pipeline" project
# Builds a GStreamer "nvivafilter" plugin with CUDA support.
# Works on host (x86_64) and for Jetson cross-compile (aarch64).
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(image_correction_pipeline LANGUAGES CXX CUDA)

# ---- Options ----------------------------------------------------------------
option(HOST_BUILD "Build on host x86_64 without Jetson sysroot" OFF)

# Force/assume CUDA toolkit root if not provided
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda CACHE PATH "CUDA Toolkit root")
endif()

# Respect user-provided arch; default to GT1030 (sm_61) for quick host tests
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61 CACHE STRING "CUDA architectures")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ---- Sources ----------------------------------------------------------------
set(CUDA_SOURCES
    src/kernel_rectify.cu
    src/wire_author_server.cu
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

# ---- Profiling-friendly compile flags  -----
# Keep CPU call stacks and CUDA line mapping visible in Nsight Systems.
target_compile_options(nvivafilter_imagecorrection PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-fno-omit-frame-pointer>
)

# Ensure NVTX markers are compiled in
target_compile_definitions(nvivafilter_imagecorrection PRIVATE NVTX_ENABLE=1)

# ---- Include directories -----------------------------------------------------
target_include_directories(nvivafilter_imagecorrection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include
)

# Only add Jetson/sysroot includes when cross-compiling
if(NOT HOST_BUILD)
    target_include_directories(nvivafilter_imagecorrection PRIVATE
        /l4t/targetfs/usr/src/jetson_multimedia_api/include
        ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/include
        ${CMAKE_SYSROOT}/usr/include
        ${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu
    )
endif()

# ---- Libraries ---------------------------------------------------------------
# CUDA runtime/driver (host first)
find_library(CUDART_LIBRARY cudart
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)
find_library(CUDA_DRIVER_LIBRARY cuda
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

# NVTX (nvToolsExt) — used by nvtxRangePush/Pop
# Host first:
find_library(NVTOOLSEXT_LIBRARY nvToolsExt
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

# If cross-compiling, also search aarch64 locations
if(NOT HOST_BUILD AND NOT NVTOOLSEXT_LIBRARY)
    find_library(NVTOOLSEXT_LIBRARY nvToolsExt
        HINTS
            ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib
            ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib/stubs
            ${CMAKE_SYSROOT}/usr/local/cuda/lib64
            ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu
    )
endif()

if(NVTOOLSEXT_LIBRARY)
    target_link_libraries(nvivafilter_imagecorrection PRIVATE ${NVTOOLSEXT_LIBRARY})
else()
    message(WARNING "nvToolsExt (NVTX) not found — NVTX markers will not link. \
Ensure CUDA Toolkit is present in the sysroot/host and try again.")
endif()

# If cross-compiling and not found yet, look in sysroot/targets too
if(NOT HOST_BUILD)
    if(NOT CUDART_LIBRARY)
        find_library(CUDART_LIBRARY cudart
            HINTS ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib
                  ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib/stubs
                  ${CMAKE_SYSROOT}/usr/local/cuda/lib64
        )
    endif()
    if(NOT CUDA_DRIVER_LIBRARY)
        find_library(CUDA_DRIVER_LIBRARY cuda
            HINTS ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib
                  ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib/stubs
                  ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
        )
    endif()
endif()

# POSIX/common
find_package(Threads REQUIRED)
find_library(LIB_DL dl)
find_library(LIB_M m)
find_library(LIB_RT rt)

# Windowing/libs:
if(HOST_BUILD)
    # On desktop: try to use system packages; all are OPTIONAL
    find_package(X11 QUIET)          # Provides X11::X11 and X11::Xext if found
    find_library(LIB_EGL EGL)        # Optional
    find_library(LIB_BSD bsd)        # Optional (libbsd-dev)
else()
    # On Jetson/sysroot: look explicitly in aarch64 paths
    find_library(LIB_EGL EGL HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
    find_library(LIB_X11 X11 HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
    find_library(LIB_XEXT Xext HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
    find_library(LIB_BSD bsd HINTS ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
endif()

# ---- Link --------------------------------------------------------------------
target_link_libraries(nvivafilter_imagecorrection
    PRIVATE
        ${CUDART_LIBRARY}
        ${CUDA_DRIVER_LIBRARY}
        Threads::Threads
)

# Host: link optional desktop libs only if found
if(HOST_BUILD)
    if(TARGET X11::X11)
        target_link_libraries(nvivafilter_imagecorrection PRIVATE X11::X11)
    endif()
    if(TARGET X11::Xext)
        target_link_libraries(nvivafilter_imagecorrection PRIVATE X11::Xext)
    endif()
    if(LIB_EGL)
        target_link_libraries(nvivafilter_imagecorrection PRIVATE ${LIB_EGL})
    endif()
    if(LIB_BSD)
        target_link_libraries(nvivafilter_imagecorrection PRIVATE ${LIB_BSD})
    endif()
    if(LIB_RT)
        target_link_libraries(nvivafilter_imagecorrection PRIVATE ${LIB_RT})
    endif()
    if(LIB_M)
        target_link_libraries(nvivafilter_imagecorrection PRIVATE ${LIB_M})
    endif()
    if(LIB_DL)
        target_link_libraries(nvivafilter_imagecorrection PRIVATE ${LIB_DL})
    endif()
else()
    # Cross-compile: assume these exist in sysroot
    if(LIB_X11)
        target_link_libraries(nvivafilter_imagecorrection PRIVATE ${LIB_X11})
    endif()
    if(LIB_XEXT)
        target_link_libraries(nvivafilter_imagecorrection PRIVATE ${LIB_XEXT})
    endif()
    if(LIB_BSD)
        target_link_libraries(nvivafilter_imagecorrection PRIVATE ${LIB_BSD})
    endif()
    if(LIB_RT)
        target_link_libraries(nvivafilter_imagecorrection PRIVATE ${LIB_RT})
    endif()
    if(LIB_M)
        target_link_libraries(nvivafilter_imagecorrection PRIVATE ${LIB_M})
    endif()
    if(LIB_DL)
        target_link_libraries(nvivafilter_imagecorrection PRIVATE ${LIB_DL})
    endif()
endif()

# ---- Sysroot link paths (only for cross) ------------------------------------
if(NOT HOST_BUILD)
    target_link_options(nvivafilter_imagecorrection PRIVATE
        -L${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu
        -L${CMAKE_SYSROOT}/lib/aarch64-linux-gnu
        -L${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib
        -L${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib/stubs
    )
endif()

# ---- Install (Jetson layout only when cross) --------------------------------
if(NOT HOST_BUILD)
    install(TARGETS nvivafilter_imagecorrection
        LIBRARY DESTINATION lib/aarch64-linux-gnu/gstreamer-1.0/
    )
endif()

# NEW: Deployment folder & run scripts
# ========================= Deploy folder & run scripts =========================
# Creates: build/image-correction-pipeline/deploy/
#   - libnvivafilter_imagecorrection.so
#   - run.sh
#   - run_profile.sh

# Where to stage deploy artifacts (on x86 build host)
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
# NEW: default sensor id
file(APPEND "${RUN_SH}" ": \${SENSOR_ID:=0}\n")
file(APPEND "${RUN_SH}" "echo \"→ Using SO: $SO\" >&2\n")
file(APPEND "${RUN_SH}" "echo \"→ Camera sensor-id=\${SENSOR_ID} → \${HOST_IP}:\${HOST_PORT} (\${WIDTH}x\${HEIGHT}@\${FPS})\" >&2\n")
file(APPEND "${RUN_SH}" "exec gst-launch-1.0 -e -v \\\n")
# NEW: pass sensor-id into nvarguscamerasrc
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
