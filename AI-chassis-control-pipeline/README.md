# Pull nvidia hoster x-compiler image
docker pull nvcr.io/nvidia/jetpack-linux-aarch64-crosscompile-x86:5.1.1

# Execute container environment
cd <"base of repo CUDA-FOR-ML-ACCELERATION">
docker run -it --rm --privileged --net=host -v /dev/bus/usb:/dev/bus/usb -v .:/workspace nvcr.io/nvidia/jetpack-linux-aarch64-crosscompile-x86:5.1.1 bash

# Execute Environment setup depending on destination SoC
If Xavier :
./setup_l4t_cross_compile_Xavier.sh

If Orin :
./setup_l4t_cross_compile_Orin.sh


# Jetson Xavier NX
cmake -S . -B build -DPROJECT=AI-chassis-control-pipeline -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/Toolchain_aarch64_l4t.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=72

cmake --build build -j"$(nproc)" --verbose

# Jetson Orin AGX
cmake -S . -B build -DPROJECT=AI-chassis-control-pipeline -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/Toolchain_aarch64_l4t.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87

cmake --build build -j"$(nproc)" --verbose