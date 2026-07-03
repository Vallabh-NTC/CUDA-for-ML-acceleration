# Pull nvidia hoster x-compiler image
`docker pull nvcr.io/nvidia/jetpack-linux-aarch64-crosscompile-x86:5.1.1`

# Execute container environment
`cd <"base of repo CUDA-FOR-ML-ACCELERATION">`
<br>
`docker run -it --rm --privileged --net=host -v /dev/bus/usb:/dev/bus/usb -v .:/workspace nvcr.io/nvidia/jetpack-linux-aarch64-crosscompile-x86:5.1.1 bash`

# Execute Environment setup depending on destination SoC

`sed -i 's/\r$//' setup_l4t_cross_compile_Xavier.sh setup_l4t_cross_compile_Orin.sh`

If Xavier :
`./setup_l4t_cross_compile_Xavier.sh`

If Orin :
`./setup_l4t_cross_compile_Orin.sh`


# Jetson Xavier NX
`cmake -S . -B build -DPROJECT=AI-chassis-control-pipeline -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/Toolchain_aarch64_l4t.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=72`

`cmake --build build -j"$(nproc)" --verbose`

# Jetson Orin AGX
`cmake -S . -B build -DPROJECT=AI-chassis-control-pipeline -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/Toolchain_aarch64_l4t.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87`

`cmake --build build -j"$(nproc)" --verbose`

---

## AI Chassis Control Pipeline (udp_decoder)

### Description
UDP/JSON decoder that integrates:
- Ethernet signals (CANape → Windows UDP Server → `udp_decoder`)
- ADMA data (INS/GNSS from iMEMS via UDP)
- Dual-camera capture (GStreamer)
- CSV logging synchronized with timestamps

### Network Configuration

- **Jetson IP**: `192.168.1.20`
- **ADMA IP (expected source)**: `192.168.1.55`
- **ETH/JSON Port**: `5005` (listening from Windows UDP Server)
- **ADMA Port**: `1021` (listening from iMEMS)

### Startup

```bash
./build/AI-chassis-control-pipeline/udp_decoder --eth-port 5005 --log_path ~/data
```

### Main Options

- `--eth-port PORT`: UDP listening port for Ethernet signals (default: 5005)
- `--log_path PATH`: directory to save images and CSV (default: ./log)

### Output

For each synchronized camera frame:
- CSV row with timestamp, ETH signals, ADMA data, image paths
- `images/cam0/` and `images/cam2/` with numbered JPEGs
- Run directory: `log/YYYYMMDD_HHMM_log_N/`