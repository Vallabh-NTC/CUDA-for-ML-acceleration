# ===== toolchain-aarch64.cmake =====
# Cross-compile for Jetson (ARM64) from x86_64 host

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# --- Sysroot ---
if(NOT DEFINED SYSROOT)
  set(SYSROOT "/l4t/targetfs")
endif()
set(CMAKE_SYSROOT "${SYSROOT}")
set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")

# Avoid try-compile running target binaries on host
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# --- Toolchain binaries ---
set(CMAKE_C_COMPILER   /l4t/toolchain/bin/aarch64-linux-gcc)
set(CMAKE_CXX_COMPILER /l4t/toolchain/bin/aarch64-linux-g++)
set(CMAKE_LINKER       /l4t/toolchain/bin/aarch64-buildroot-linux-gnu-ld)

set(CMAKE_AR      /l4t/toolchain/bin/aarch64-buildroot-linux-gnu-ar)
set(CMAKE_NM      /l4t/toolchain/bin/aarch64-buildroot-linux-gnu-nm)
set(CMAKE_RANLIB  /l4t/toolchain/bin/aarch64-buildroot-linux-gnu-ranlib)

# --- CUDA ---
# Use host nvcc (x86) with host g++ for preprocessing.
# Linking still uses cross g++ (above).
if(EXISTS "/usr/local/cuda-11.4/bin/nvcc")
  set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.4/bin/nvcc CACHE FILEPATH "" FORCE)
elseif(EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc CACHE FILEPATH "" FORCE)
endif()

# Host compiler for nvcc = host g++
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++ CACHE FILEPATH "" FORCE)

# Do NOT inject --sysroot into nvcc (breaks preprocessing)
set(CMAKE_CUDA_FLAGS_INIT "")

# --- Default compile/link flags ---
set(_SYS_INC "${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu")
set(_SYS_LIB1 "${CMAKE_SYSROOT}/usr/local/cuda-11.4/targets/aarch64-linux/lib")
set(_SYS_LIB2 "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu")
set(_SYS_LIB3 "${CMAKE_SYSROOT}/lib/aarch64-linux-gnu")

set(CMAKE_C_FLAGS_INIT   "--sysroot=${CMAKE_SYSROOT} -I${_SYS_INC}")
set(CMAKE_CXX_FLAGS_INIT "--sysroot=${CMAKE_SYSROOT} -I${_SYS_INC}")

set(CMAKE_EXE_LINKER_FLAGS_INIT
    "--sysroot=${CMAKE_SYSROOT} -L${_SYS_LIB1} -L${_SYS_LIB2} -L${_SYS_LIB3} -Wl,-rpath-link,${_SYS_LIB1} -Wl,-rpath-link,${_SYS_LIB2}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT
    "--sysroot=${CMAKE_SYSROOT} -L${_SYS_LIB1} -L${_SYS_LIB2} -L${_SYS_LIB3} -Wl,-rpath-link,${_SYS_LIB1} -Wl,-rpath-link,${_SYS_LIB2}")

# --- CMake search policy ---
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# --- Threads fix for cross env ---
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(Threads_FOUND TRUE)

# --- pkg-config inside sysroot ---
set(ENV{PKG_CONFIG_DIR} "")
set(ENV{PKG_CONFIG_LIBDIR} "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig:${CMAKE_SYSROOT}/usr/share/pkgconfig:${CMAKE_SYSROOT}/usr/lib/pkgconfig")
set(ENV{PKG_CONFIG_SYSROOT_DIR} "${CMAKE_SYSROOT}")

# --- Skip CUDA compiler tests (cross-compile environment) ---
set(CMAKE_CUDA_COMPILER_WORKS TRUE CACHE INTERNAL "")
set(CMAKE_CUDA_COMPILER_FORCED TRUE CACHE INTERNAL "")
