#!/bin/bash
# CUDA FGN Kernel Build Script
# Requires: CUDA Toolkit with nvcc, cuFFT, and cuRAND

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building CUDA FGN kernel..."

# Detect GPU architecture (optional, defaults to common architectures)
# You can specify your GPU's compute capability for better performance
# Common values: sm_75 (Turing), sm_80 (Ampere), sm_86 (Ampere), sm_89 (Ada Lovelace)
ARCH="${CUDA_ARCH:-sm_75}"

echo "Target architecture: $ARCH"

# Build for Linux
nvcc -O3 -use_fast_math -arch=$ARCH \
    -shared fgn_exports.cu \
    -o ./fgn_linux/libfgn.so \
    -Xcompiler -fPIC \
    -lcufft -lcurand

echo "Built: fgn_linux/libfgn.so"
echo "Done!"
