# CUDA FGN Kernel Build Instructions

This directory contains the CUDA implementation of Fractional Gaussian Noise (FGN) generation using the circulant embedding method with FFT.

## Requirements

- NVIDIA GPU with Compute Capability 7.5+ (Turing or newer recommended)
- CUDA Toolkit 11.0+ (with nvcc, cuFFT, and cuRAND)
- For Windows: Visual Studio 2019/2022 with C++ build tools

## Quick Build (automatic via Cargo)

When building with the `cuda` feature, `build.rs` now tries to compile `fgn_exports.cu` automatically:

```bash
cargo build --features cuda
```

Environment variables:

- `CUDA_ARCH` (default: `sm_75`)
- `CUDA_NVCC` (default: `nvcc`)
- `STOCHASTIC_RS_SKIP_CUDA_BUILD=1` to skip auto-build
- `STOCHASTIC_RS_CUDA_FGN_LIB_PATH=/path/to/libfgn_cuda.so|fgn_cuda.dll` to force a runtime library path

## Manual Build

### Linux
```bash
cd src/stochastic/cuda
chmod +x build.sh
./build.sh
```

### Windows
```cmd
cd src\stochastic\cuda
build.bat
```

## Manual Build Commands

### Linux
```bash
nvcc -O3 -use_fast_math -arch=sm_75 \
    -shared fgn_exports.cu \
    -o ./fgn_linux/libfgn.so \
    -Xcompiler -fPIC \
    -lcufft -lcurand
```

### Windows (PowerShell)
```powershell
$env:TEMP = "C:\Temp"
$env:TMP = "C:\Temp"
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && nvcc -O3 -use_fast_math -arch=sm_89 -shared fgn_exports.cu -o ./fgn_windows/fgn.dll -lcufft -lcurand'
```

## GPU Architecture

Set the `-arch` flag to match your GPU's compute capability:

| GPU Generation | Compute Capability | Flag |
|---------------|-------------------|------|
| Turing (RTX 20xx) | 7.5 | `-arch=sm_75` |
| Ampere (RTX 30xx) | 8.6 | `-arch=sm_86` |
| Ampere (A100) | 8.0 | `-arch=sm_80` |
| Ada Lovelace (RTX 40xx) | 8.9 | `-arch=sm_89` |
| Hopper (H100) | 9.0 | `-arch=sm_90` |

You can find your GPU's compute capability with:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

## Usage in Rust

Enable the `cuda` feature in your `Cargo.toml`:
```toml
[dependencies]
stochastic-rs = { version = "1.1.1", features = ["cuda"] }
```

Then use `sample_cuda()` instead of `sample()`:
```rust
use stochastic_rs::stochastic::noise::fgn::FGN;
use stochastic_rs::stochastic::SamplingExt;

let fgn = FGN::new(0.7_f64, 1000, Some(1.0));
let samples = fgn.sample_cuda().unwrap();
```

CUDA sampling now supports both `FGN<f32>` and `FGN<f64>` backends when the compiled CUDA library exports `fgn32_*` and `fgn64_*`.

## Performance

CUDA acceleration provides 2-3x speedup for batch sampling:

| Operation | CPU (ms) | CUDA (ms) | Speedup |
|-----------|----------|-----------|---------|
| 1000 repeated samples | 80 | 40 | 2x |
| 10000 batch | 91 | 33 | 2.8x |
| 100000 batch | 854 | 328 | 2.6x |

## Troubleshooting

### Windows: "nvcc fatal : Cannot find compiler 'cl.exe'"
Run the build from a Visual Studio Developer Command Prompt, or use `build.bat` which sets up the environment.

### Windows: Path with special characters
The build scripts set `TEMP` and `TMP` to `C:\Temp` to avoid issues with usernames containing special characters.

### Linux: "libcufft.so not found"
Ensure CUDA libraries are in your `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
