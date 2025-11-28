#!/usr/bin/env bash
set -e
set -o pipefail

echo "=========================================="
echo "  L4T Cross-Compile Setup (t186ref)"
echo "=========================================="

# -------------------------------
# CONFIG
# -------------------------------

PUBLIC_DIR="/l4t_sources"
TOOLCHAIN_PREFIX="/l4t/toolchain/bin/aarch64-buildroot-linux-gnu-"

echo "[INFO] Public sources dir: $PUBLIC_DIR"
echo "[INFO] Toolchain prefix  : $TOOLCHAIN_PREFIX"

# ============================================================
# 1) UNPACK TARGET ROOTFS + TOOLCHAIN INTO /l4t
# ============================================================

echo "[STEP 1] Extracting targetfs..."
cd /l4t
tar -I lbzip2 -xf targetfs.tbz2

echo "[STEP 1] Extracting toolchain..."
mkdir -p toolchain
tar -C toolchain -xf toolchain.tar.gz

# ============================================================
# 2) EXTRACT PUBLIC SOURCES INTO /l4t_sources
# ============================================================

echo "[STEP 2] Creating public source directory: $PUBLIC_DIR"
mkdir -p $PUBLIC_DIR

echo "[STEP 2] Extracting public sources..."
cd $PUBLIC_DIR
tar -I lbzip2 -xf /workspace/public_sources.tbz2

echo "[STEP 2] Entering public source root..."
cd $PUBLIC_DIR/Linux_for_Tegra/source/public

# ============================================================
# 3) BUILD NVIDIA PUBLIC SOURCES (Xavier NX → t186ref)
# ============================================================

echo "[STEP 3] Building nv_public_src (t186ref)..."

export CROSS_COMPILE_AARCH64=$TOOLCHAIN_PREFIX
export CROSS_COMPILE_AARCH64_PATH=/l4t/toolchain
export NV_TARGET_BOARD=t186ref

./nv_public_src_build.sh

# ============================================================
# 4) FIX CRT STARTUP OBJECTS
# ============================================================

echo "[STEP 4] Fixing crt startup objects..."

ln -sf /l4t/targetfs/usr/lib/aarch64-linux-gnu/crt1.o /l4t/targetfs/lib/crt1.o
ln -sf /l4t/targetfs/usr/lib/aarch64-linux-gnu/crti.o /l4t/targetfs/lib/crti.o
ln -sf /l4t/targetfs/usr/lib/aarch64-linux-gnu/crtn.o /l4t/targetfs/lib/crtn.o

# ============================================================
# 5) FIX GLIBC INCLUDE PATHS
# ============================================================

echo "[STEP 5] Fixing glibc include dirs..."

ln -sfn /l4t/targetfs/usr/include/aarch64-linux-gnu/sys /l4t/targetfs/usr/include/sys
ln -sfn /l4t/targetfs/usr/include/aarch64-linux-gnu/gnu /l4t/targetfs/usr/include/gnu
ln -sfn /l4t/targetfs/usr/include/aarch64-linux-gnu/asm /l4t/targetfs/usr/include/asm

# ============================================================
# 6) FIX X11 / LIBBSD SYMLINKS
# ============================================================

echo "[STEP 6] Fixing X11 + libbsd symlinks..."

ln -sf /l4t/targetfs/usr/lib/aarch64-linux-gnu/libXext.so.6 /l4t/targetfs/usr/lib/aarch64-linux-gnu/libXext.so
ln -sf /l4t/targetfs/usr/lib/aarch64-linux-gnu/libbsd.so.0  /l4t/targetfs/usr/lib/aarch64-linux-gnu/libbsd.so

# ============================================================
# 7) WRAP NVCC (strip unsupported flags)
# ============================================================

echo "[STEP 7] Wrapping nvcc..."

if [[ -f /usr/local/cuda/bin/nvcc ]]; then
    mv /usr/local/cuda/bin/nvcc /usr/local/cuda/bin/nvcc.real
fi

tee /usr/local/cuda/bin/nvcc >/dev/null <<'SH'
#!/usr/bin/env bash
real_nvcc=/usr/local/cuda/bin/nvcc.real
args=()
skip_next=0
for a in "$@"; do
    if [[ $skip_next -eq 1 ]]; then skip_next=0; continue; fi
    case "$a" in
        --target-dir) skip_next=1; continue ;;
        --sysroot=*) continue ;;
        *) args+=("$a") ;;
    esac
done
exec "$real_nvcc" "${args[@]}"
SH

chmod +x /usr/local/cuda/bin/nvcc

# ============================================================
# 8) INSTALL / UPGRADE CMAKE (to 3.22.6)
# ============================================================

echo "[STEP 8] Upgrading CMake to 3.22.6..."

apt-get update
apt-get install -y wget build-essential libssl-dev

wget https://cmake.org/files/v3.22/cmake-3.22.6-linux-x86_64.tar.gz

tar -xzf cmake-3.22.6-linux-x86_64.tar.gz -C /usr/local --strip-components=1

echo "[INFO] CMake version:"
cmake --version

echo "=========================================="
echo "  L4T CROSS-COMPILE ENVIRONMENT READY"
echo "  Board: t186ref (Xavier NX)"
echo "  Sources: $PUBLIC_DIR"
echo "=========================================="
