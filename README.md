# Pull nvidia hoster x-compiler image
`docker pull nvcr.io/nvidia/jetpack-linux-aarch64-crosscompile-x86:5.1.1`

# Execute container environment
`cd <"base of repo CUDA-FOR-ML-ACCELERATION">`

Option 1: <br>
Download manually this file : 
https://developer.nvidia.com/embedded/l4t/r35_release_v1.0/sources/public_sources.tbz2

`docker run -it --rm --privileged --net=host -v /dev/bus/usb:/dev/bus/usb -v .:/workspace nvcr.io/nvidia/jetpack-linux-aarch64-crosscompile-x86:5.1.1 bash`

Option 2: <br>
Docker pull complete image from gitlab registry (need to request access)
cr.cicd.skyway.porsche.com/mvmkr25/moviemaker_backend/jetpack-xcc:5.1.1-custom

# Execute Environment setup depending on destination SoC
If Xavier :
`./setup_l4t_cross_compile_Xavier.sh`

If Orin :
`./setup_l4t_cross_compile_Orin.sh`

# Build individual/specific projects for dedicated SoC

For example if one wants to build tha "AI-chassis-control-pipeline" project for both Jetson Xavier :

`cmake -S . -B build -DPROJECT=AI-chassis-control-pipeline -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/Toolchain_aarch64_l4t.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=72`

`cmake --build build -j"$(nproc)" --verbose`