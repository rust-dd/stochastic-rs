![Build Workflow](https://github.com/dancixx/stochastic-rs/actions/workflows/rust.yml/badge.svg)
[![Crates.io](https://img.shields.io/crates/v/stochastic-rs?style=flat-square)](https://crates.io/crates/stochastic-rs)
![License](https://img.shields.io/crates/l/stochastic-rs?style=flat-square)
[![codecov](https://codecov.io/gh/dancixx/stochastic-rs/graph/badge.svg?token=SCSp3z7BQJ)](https://codecov.io/gh/dancixx/stochastic-rs)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fdancixx%2Fstochastic-rs.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fdancixx%2Fstochastic-rs?ref=badge_shield)

# stochastic-rs

A high-performance Rust library for simulating stochastic processes, with first-class bindings. Built for quantitative finance, statistical modeling and synthetic data generation.

## Features

- **85+ stochastic models** - 31 diffusions (OU, CIR, GBM, CEV, CKLS, Aït-Sahalia, Pearson, Jacobi, regime-switching, ...), 15 jump processes (Merton, Kou, CGMY, bilateral gamma, ...), 9 stochastic volatility models (Heston, SABR, Bergomi, rough Bergomi, HKDE, ...), 13 interest rate models (Hull-White, HJM, Vasicek, ...), and base processes (fBM, Poisson, LFSM, ...)
- **MLE engine** - maximum likelihood estimation for 1-D diffusion models with 6 transition density approximations (Euler, Ozaki, Shoji-Ozaki, Elerian, Kessler, Aït-Sahalia), L-BFGS optimizer via argmin, and 22 built-in process implementations
- **Quant toolbox** - option pricing (Fourier, barrier, lookback, Asian, variance swaps, regime-switching), calibration (Heston, SABR, Lévy, SVJ, rough Bergomi), and finite-difference methods
- **Copulas** - bivariate, multivariate, and empirical copulas with correlation utilities
- **Statistics** - kernel density estimation, fractional OU estimation, and CIR parameter fitting
- **SIMD-optimized** - fractional Gaussian noise, fractional Brownian motion, and all probability distributions use wide SIMD for fast sample generation
- **Parallel sampling** - `sample_par(m)` generates `m` independent paths in parallel via rayon
- **Generic precision** - most models support both `f32` and `f64`
- **Python bindings** - full stochastic model coverage with numpy integration; all models return numpy arrays (others coming soon)

## Installation

### Rust

```toml
[dependencies]
stochastic-rs = "1.5.0"
```

### Bindings

```bash
pip install stochastic-rs
```

For development builds from source (requires [maturin](https://www.maturin.rs/)):

```bash
pip install maturin
maturin develop --release
```

### OpenBLAS (required for `openblas` feature)

The `openblas` feature enables `ndarray-linalg` for linear algebra operations. It requires a system OpenBLAS installation with LAPACK support.

**Linux (Debian/Ubuntu)**

```bash
sudo apt install libopenblas-dev
```

**Linux (Fedora/RHEL)**

```bash
sudo dnf install openblas-devel
```

**macOS**

```bash
brew install openblas
export OPENBLAS_DIR=$(brew --prefix openblas)
```

**Windows**

Download prebuilt OpenBLAS from [OpenMathLib/OpenBLAS releases](https://github.com/OpenMathLib/OpenBLAS/releases) (pick the `x64.zip`), extract it, and install [vcpkg](https://github.com/microsoft/vcpkg):

```powershell
git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
$env:VCPKG_ROOT = "C:\vcpkg"
```

Then copy the prebuilt `libopenblas.lib` and `libopenblas.dll` into `$VCPKG_ROOT\installed\x64-windows\lib\` and `$VCPKG_ROOT\installed\x64-windows\bin\` respectively. The prebuilt release includes LAPACK (the vcpkg `openblas` port does not).

**Build with OpenBLAS**

```bash
cargo build --features openblas
```

### CUDA native (optional)

Requires NVIDIA CUDA Toolkit (12.x+) and a compatible GPU.

```bash
cargo build --features cuda-native
```

## Usage

### Rust

```rust
use stochastic_rs::stochastic::process::fbm::FBM;
use stochastic_rs::stochastic::volatility::heston::Heston;
use stochastic_rs::stochastic::volatility::HestonPow;
use stochastic_rs::traits::ProcessExt;

fn main() {
    // Fractional Brownian Motion
    let fbm = FBM::new(0.7, 1000, None);
    let path = fbm.sample();

    // Parallel batch sampling
    let paths = fbm.sample_par(1000);

    // Heston stochastic volatility
    let heston = Heston::new(
        Some(100.0),   // s0
        Some(0.04),    // v0
        2.0,           // kappa
        0.04,          // theta
        0.3,           // sigma
        -0.7,          // rho
        0.05,          // mu
        1000,          // n
        None,          // t
        HestonPow::Sqrt,
        Some(false),
    );
    let [price, variance] = heston.sample();
}
```

### Bindings

All models return numpy arrays. Use `dtype="f32"` or `dtype="f64"` (default) to control precision.

```python
import stochastic_rs as sr

# Basic processes
fbm = sr.PyFBM(0.7, 1000)
path = fbm.sample()           # shape (1000,)
paths = fbm.sample_par(500)   # shape (500, 1000)

# Stochastic volatility
heston = sr.PyHeston(mu=0.05, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, n=1000)
price, variance = heston.sample()

# Models with callable parameters
hw = sr.PyHullWhite(theta=lambda t: 0.04 + 0.01*t, alpha=0.1, sigma=0.02, n=1000)
rates = hw.sample()

# Jump processes with custom jump distributions
import numpy as np
merton = sr.PyMerton(
    alpha=0.05, sigma=0.2, lambda_=3.0, theta=0.01,
    distribution=lambda: np.random.normal(0, 0.1),
    n=1000,
)
log_prices = merton.sample()
```

## Benchmarks

### FGN CPU vs CUDA native (`sample`, `sample_par`, `sample_cuda_native`)

`cuda-native` backend: cudarc + cuFFT + fused Philox RNG kernel (no `.cu` files, no `nvcc`).

```bash
cargo bench --features cuda-native --bench fgn_cuda_native
```

Environment: NVIDIA GPU, CUDA 12.x, Rust nightly, `--release` with LTO.

Single path (`sample` vs `sample_cuda_native(1)`, `f32`, H=0.7):

| n | CPU `sample` | CUDA `sample_cuda_native(1)` | Speedup |
|---:|---:|---:|---:|
| 1,024 | 8.1 us | 46 us | 0.18x |
| 4,096 | 35 us | 84 us | 0.42x |
| 16,384 | 147 us | 110 us | **1.3x** |
| 65,536 | 850 us | 227 us | **3.7x** |

Batch (`sample_par(m)` vs `sample_cuda_native(m)`, `f32`, H=0.7):

| n, m | CPU `sample_par(m)` | CUDA `sample_cuda_native(m)` | Speedup |
|---|---:|---:|---:|
| 4,096, 32 | 147 us | 117 us | **1.3x** |
| 4,096, 512 | 1.78 ms | 2.37 ms | 0.75x |
| 65,536, 128 | 12.6 ms | 10.5 ms | **1.2x** |
| 65,536, 1024 | 102 ms | 93 ms | **1.1x** |

CUDA wins for large n (>= 16k) and is competitive at n=65k batches. CPU rayon parallelism dominates for medium n due to zero transfer overhead.

### Distribution Sampling (All Built-in Distributions)

Measured with:

```bash
cargo bench --bench dist_multicore
```

Configuration in this run:
- `sample_matrix` benchmark
- 1-thread vs 14-thread rayon pools
- size is mostly `1024 x 1024`; heavy discrete samplers use `512 x 512`

| Distribution | Shape | 1T (ms) | MT (ms) | Speedup |
|---|---:|---:|---:|---:|
| Normal<f64> | 1024 x 1024 | 1.78 | 0.34 | 5.28x |
| Exp<f64> | 1024 x 1024 | 1.73 | 0.33 | 5.25x |
| Uniform<f64> | 1024 x 1024 | 0.65 | 0.13 | 5.12x |
| Cauchy<f64> | 1024 x 1024 | 6.23 | 0.90 | 6.96x |
| LogNormal<f64> | 1024 x 1024 | 5.07 | 0.81 | 6.25x |
| Gamma<f64> | 1024 x 1024 | 5.20 | 0.72 | 7.19x |
| ChiSq<f64> | 1024 x 1024 | 5.06 | 1.22 | 4.14x |
| StudentT<f64> | 1024 x 1024 | 7.89 | 1.89 | 4.18x |
| Beta<f64> | 1024 x 1024 | 11.85 | 1.68 | 7.04x |
| Weibull<f64> | 1024 x 1024 | 13.17 | 1.73 | 7.59x |
| Pareto<f64> | 1024 x 1024 | 5.48 | 0.80 | 6.87x |
| InvGauss<f64> | 1024 x 1024 | 2.52 | 0.44 | 5.69x |
| NIG<f64> | 1024 x 1024 | 5.93 | 0.90 | 6.62x |
| AlphaStable<f64> | 1024 x 1024 | 42.52 | 5.36 | 7.94x |
| Poisson<i64> | 1024 x 1024 | 2.28 | 0.42 | 5.40x |
| Geometric<u64> | 1024 x 1024 | 2.75 | 0.44 | 6.30x |
| Binomial<u32> | 512 x 512 | 4.43 | 0.70 | 6.32x |
| Hypergeo<u32> | 512 x 512 | 20.99 | 2.76 | 7.60x |

`Normal` single-thread kernel comparison (`fill_slice`, same run):
- vs `rand_distr + SimdRng`: ~`1.21x` to `1.35x`
- vs `rand_distr + rand::rng()`: ~`4.09x` to `4.61x`

## Contributing

Contributions are welcome - bug reports, feature suggestions, or PRs. Open an issue or start a discussion on GitHub.

## License

MIT - see [LICENSE](https://github.com/dancixx/stochastic-rs/blob/main/LICENSE).
