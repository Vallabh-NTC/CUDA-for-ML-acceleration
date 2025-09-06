README — Cross Compilation of image-correction-pipeline for Jetson (JetPack 5.1.1)

🎯 Objective

This project implements an image correction GStreamer pipeline using the nvivafilter plugin with CUDA kernels.
The main goals are:

Accelerate image rectification directly on the GPU of Jetson devices.

Use CUDA NVVM and zero-copy processing with GStreamer.

Cross-compile the project on an x86 host using NVIDIA’s JetPack cross-compilation container for JetPack 5.1.1 (L4T R35.3.1).

🛠️ Cross-Compilation Environment

We rely on NVIDIA’s official JetPack Linux Cross-Compilation Docker container:

docker pull nvcr.io/nvidia/jetpack-linux-aarch64-crosscompile-x86:5.1.1


Run the container:

docker run -it --privileged --net=host \
  -v /dev/bus/usb:/dev/bus/usb \
  -v ${WORKSPACE}:/workspace \
  nvcr.io/nvidia/jetpack-linux-aarch64-crosscompile-x86:5.1.1


Inside the container, extract the sysroot and toolchain provided in /l4t:

cd /l4t
cat targetfs.tbz2.* > targetfs.tbz2
tar -I lbzip2 -xf targetfs.tbz2

mkdir toolchain
tar -C toolchain -xf toolchain.tar.gz


Now you have:

Sysroot under /l4t/targetfs

Toolchain under /l4t/toolchain

To cross compile L4T public sources

Download L4T Driver Package (BSP) Sources which should be named as public_source.tbz2 from Jetson Linux page and place it under ${WORKSPACE} before starting the container.
Inside the container run the commands below:
JetPack 5
cd /workspace
tar -I lbzip2 -xf public_sources.tbz2
cd ./Linux_for_Tegra/source/public
CROSS_COMPILE_AARCH64=/l4t/toolchain/bin/aarch64-buildroot-linux-gnu- CROSS_COMPILE_AARCH64_PATH=/l4t/toolchain NV_TARGET_BOARD=t186ref ./nv_public_src_build.sh

To cross compile Jetson Multimedia API samples (JetPack 5 only)
Inside the container run the commands below:
cd /l4t/targetfs/usr/src/jetson_multimedia_api
export CROSS_COMPILE=aarch64-linux-gnu-
export TARGET_ROOTFS=/l4t/targetfs/
make


🧩 Common Errors & Fixes
1. crt1.o: No such file or directory

Error:

/l4t/toolchain/bin/aarch64-buildroot-linux-gnu-gcc: fatal error: cannot find crt1.o


Fix: Create symlinks inside sysroot:

ln -s /l4t/targetfs/usr/lib/aarch64-linux-gnu/crt1.o /l4t/targetfs/lib/
ln -s /l4t/targetfs/usr/lib/aarch64-linux-gnu/crti.o /l4t/targetfs/lib/
ln -s /l4t/targetfs/usr/lib/aarch64-linux-gnu/crtn.o /l4t/targetfs/lib/

2. fatal error: sys/cdefs.h: No such file or directory

Error:

/l4t/targetfs/usr/include/features.h:461:12: fatal error: sys/cdefs.h: No such file or directory


Fix: The header files from NVIDIA Multimedia API need to be available. Copy them into sysroot includes:

cp -r /l4t/targetfs/usr/src/jetson_multimedia_api/include/* /l4t/targetfs/usr/include/

3. CUDA_ARCHITECTURES is empty

Error:

CMake Error: CUDA_ARCHITECTURES is empty for target "nvivafilter_rectify".


Fix: Explicitly set the architecture in your CMake command or in CMakeLists.txt:

cmake .. -DCMAKE_CUDA_ARCHITECTURES=72

4. Old CMake version (3.16)

CMake 3.16 (default in Ubuntu 20.04) does not fully support CUDA cross-compilation.

Fix: Install a newer CMake (≥ 3.18, ideally ≥ 3.24):

sudo apt remove --purge cmake
sudo apt install -y apt-transport-https ca-certificates gnupg software-properties-common wget

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
  | gpg --dearmor - \
  | sudo tee /usr/share/keyrings/kitware.gpg >/dev/null

echo "deb [signed-by=/usr/share/keyrings/kitware.gpg] https://apt.kitware.com/ubuntu/ focal main" \
  | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null

sudo apt update
sudo apt install cmake
cmake --version

5. nvcc fatal: A single input file is required for a non-link phase when an outputfile is specified

What it means
This error appears misleading: it is not about missing .cu files.
It occurs because the host nvcc (x86) is invoked with flags (--sysroot, --target-dir) that are only valid on Jetson’s native nvcc (aarch64).
The host nvcc cannot parse them → it misinterprets arguments and fails with this generic “single input file” message.

Why it happens in cross-compilation

The JetPack toolchain injects --sysroot=/l4t/targetfs and --target-dir aarch64-linux.

These are accepted on Jetson, but not on x86 host nvcc.

Fix
We implemented an nvcc wrapper that strips problematic flags and forwards everything else to the real nvcc.

Move the original nvcc:

sudo mv /usr/local/cuda/bin/nvcc /usr/local/cuda/bin/nvcc.real


Create wrapper /usr/local/cuda/bin/nvcc:

#!/bin/bash
real_nvcc=/usr/local/cuda/bin/nvcc.real
args=(); skip_next=0
for arg in "$@"; do
  if [[ $skip_next -eq 1 ]]; then skip_next=0; continue; fi
  case "$arg" in
    --sysroot=*)   continue ;;
    --target-dir)  skip_next=1; continue ;;
    --target-dir*) continue ;;
    *)             args+=("$arg") ;;
  esac
done
exec "$real_nvcc" "${args[@]}"


Make it executable:

sudo chmod +x /usr/local/cuda/bin/nvcc


✅ Result: CMake now invokes the wrapper transparently, the invalid flags are filtered out, and nvcc works as expected.

6. Missing standard headers in sysroot

Errors observed

fatal error: sys/cdefs.h: No such file or directory
fatal error: bits/wordsize.h: No such file or directory
fatal error: gnu/stubs.h: No such file or directory
fatal error: asm/errno.h: No such file or directory


Cause
Some libc headers exist only under /usr/include/aarch64-linux-gnu/ inside the sysroot.
CMake was not searching these paths, or symlinks were missing.

Fix

Verified the files with find:

/l4t/targetfs/usr/include/aarch64-linux-gnu/sys/cdefs.h

/l4t/targetfs/usr/include/aarch64-linux-gnu/bits/wordsize.h

/l4t/targetfs/usr/include/aarch64-linux-gnu/gnu/stubs.h

/l4t/targetfs/usr/include/aarch64-linux-gnu/asm/errno.h

Added include paths in image-correction-pipeline/CMakeLists.txt:

target_include_directories(nvivafilter_rectify PRIVATE
    ${CMAKE_SYSROOT}/usr/include
    ${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu
)


Fixed broken symlinks if necessary:

ln -s /l4t/targetfs/usr/include/aarch64-linux-gnu/sys /l4t/targetfs/usr/include/sys
ln -s /l4t/targetfs/usr/include/aarch64-linux-gnu/gnu /l4t/targetfs/usr/include/gnu
ln -s /l4t/targetfs/usr/include/aarch64-linux-gnu/asm /l4t/targetfs/usr/include/asm


✅ Result: the compiler successfully found glibc headers under the sysroot.

7. CMake ignoring wrapper and still detecting /usr/local/cuda/bin/nvcc

Symptom
Even after specifying -DCMAKE_CUDA_COMPILER=/usr/local/cuda-wrapper/bin/nvcc,
CMakeCache.txt still contained:

CMAKE_CUDA_COMPILER:STRING=/usr/local/cuda/bin/nvcc


Cause
CMake always probes /usr/local/cuda/bin/nvcc during compiler detection.

Fix
We forced it to pick the wrapper by replacing the binary (NVIDIA suggestion):

Renamed original nvcc → nvcc.real

Put wrapper script in its place as /usr/local/cuda/bin/nvcc.

✅ Result: CMake cannot bypass the wrapper anymore — every nvcc invocation goes through our filter.

8. fatal error: X11/Xlib.h: No such file or directory

Error:

/l4t/targetfs/usr/src/jetson_multimedia_api/include/EGL/eglplatform.h:134:10: fatal error: X11/Xlib.h: No such file or directory


Cause:
The EGL headers (eglplatform.h) inside Jetson Multimedia API assume an X11 environment and try to include X11/Xlib.h.
On the host sysroot, X11 headers and libraries were present but CMake/nvcc was not including them correctly.

Fix:
We confirmed that the libraries existed in the sysroot:

/l4t/targetfs/usr/lib/aarch64-linux-gnu/libX11.so
/l4t/targetfs/usr/lib/aarch64-linux-gnu/libxcb.so
/l4t/targetfs/usr/lib/aarch64-linux-gnu/libXau.so
/l4t/targetfs/usr/lib/aarch64-linux-gnu/libXdmcp.so


Then we explicitly passed them to the build system:

-DCMAKE_EXE_LINKER_FLAGS="--sysroot=/l4t/targetfs \
   -L/l4t/targetfs/lib/aarch64-linux-gnu \
   -L/l4t/targetfs/usr/lib/aarch64-linux-gnu \
   -lX11 -lxcb -lXau -lXdmcp -ldl -lbsd" \
-DCMAKE_SHARED_LINKER_FLAGS="--sysroot=/l4t/targetfs \
   -L/l4t/targetfs/lib/aarch64-linux-gnu \
   -L/l4t/targetfs/usr/lib/aarch64-linux-gnu \
   -lX11 -lxcb -lXau -lXdmcp -ldl -lbsd"


✅ Result: EGL/X11 dependencies were found and properly linked.

9. Linker errors: cannot find /lib64/libc.so.6 and missing ld-linux-aarch64.so.1

Error:

/l4t/toolchain/.../ld: cannot find /lib64/libc.so.6
/l4t/toolchain/.../ld: cannot find /usr/lib64/libc_nonshared.a
/l4t/toolchain/.../ld: cannot find /lib/ld-linux-aarch64.so.1


Cause:
The linker was searching in host /lib64 instead of inside the Jetson sysroot.

Fix:
We forced the linker to always use the correct sysroot:

-DCMAKE_EXE_LINKER_FLAGS="--sysroot=/l4t/targetfs \
   -L/l4t/targetfs/lib/aarch64-linux-gnu \
   -L/l4t/targetfs/usr/lib/aarch64-linux-gnu" \
-DCMAKE_SHARED_LINKER_FLAGS="--sysroot=/l4t/targetfs \
   -L/l4t/targetfs/lib/aarch64-linux-gnu \
   -L/l4t/targetfs/usr/lib/aarch64-linux-gnu"


✅ Result: Linking now correctly finds libc.so, ld-linux-aarch64.so.1, and other runtime libraries inside the sysroot.

10. Additional X11 dependencies (libdl, libbsd)

Error:
During the first -lX11 test link, the linker complained about missing symbols:

undefined reference to `dlsym@GLIBC_2.17`
undefined reference to `arc4random_buf@LIBBSD_0.2`


Cause:
libX11 depends on libdl and libbsd. These libraries were present in the sysroot but not linked explicitly.

Fix:
We added them to linker flags:

-lX11 -lxcb -lXau -lXdmcp -ldl -lbsd


✅ Result: Link succeeded, all symbols resolved.

11. ✅ Final CMake Command

The working cross-compilation command is:

rm -rf build/*

cmake -S . -B build \
  -DPROJECT=image-correction-pipeline \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/Toolchain_aarch64_l4t.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=72 \
  -DCMAKE_CUDA_FLAGS="--compiler-options=-I/l4t/targetfs/usr/include \
                      --compiler-options=-I/l4t/targetfs/usr/include/aarch64-linux-gnu" \
  -DCMAKE_EXE_LINKER_FLAGS="--sysroot=/l4t/targetfs \
      -L/l4t/targetfs/lib/aarch64-linux-gnu \
      -L/l4t/targetfs/usr/lib/aarch64-linux-gnu \
      -lX11 -lxcb -lXau -lXdmcp -ldl -lbsd" \
  -DCMAKE_SHARED_LINKER_FLAGS="--sysroot=/l4t/targetfs \
      -L/l4t/targetfs/lib/aarch64-linux-gnu \
      -L/l4t/targetfs/usr/lib/aarch64-linux-gnu \
      -lX11 -lxcb -lXau -lXdmcp -ldl -lbsd"


Build with:

make -C build -j$(nproc) VERBOSE=1

12. Verification of the final binary

Check architecture:

file build/image-correction-pipeline/libnvivafilter_rectify.so
# → ELF 64-bit LSB shared object, ARM aarch64


Check linked libraries:

aarch64-linux-gnu-objdump -p build/image-correction-pipeline/libnvivafilter_rectify.so | grep NEEDED


Output:

NEEDED               libX11.so.6
NEEDED               libxcb.so.1
NEEDED               libXau.so.6
NEEDED               libXdmcp.so.6
NEEDED               libdl.so.2
NEEDED               libbsd.so.0
NEEDED               libcudart.so.11.0
NEEDED               libcuda.so.1
NEEDED               librt.so.1
NEEDED               libpthread.so.0
NEEDED               libstdc++.so.6
NEEDED               libm.so.6
NEEDED               libgcc_s.so.1
NEEDED               libc.so.6


✅ Result:

The shared library is correctly cross-compiled for ARM aarch64.

All CUDA, EGL, and X11 dependencies are properly linked.

Ready to deploy on Jetson Xavier/Nano/TX2