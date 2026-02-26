![Build Workflow](https://github.com/dancixx/stochastic-rs/actions/workflows/rust.yml/badge.svg)
[![Crates.io](https://img.shields.io/crates/v/stochastic-rs?style=flat-square)](https://crates.io/crates/stochastic-rs)
![License](https://img.shields.io/crates/l/stochastic-rs?style=flat-square)
[![codecov](https://codecov.io/gh/dancixx/stochastic-rs/graph/badge.svg?token=SCSp3z7BQJ)](https://codecov.io/gh/dancixx/stochastic-rs)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fdancixx%2Fstochastic-rs.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fdancixx%2Fstochastic-rs?ref=badge_shield)

# stochastic-rs

A high-performance Rust library for simulating stochastic processes, with first-class bindings. Built for quantitative finance, statistical modeling and synthetic data generation.

## Features

- **85+ stochastic models** - diffusions, jump processes, stochastic volatility, interest rate models, autoregressive models, noise generators, and probability distributions
- **Copulas** - bivariate, multivariate, and empirical copulas with correlation utilities
- **Quant toolbox** - option pricing, bond analytics, calibration, loss models, order book, and trading strategies
- **Statistics** - MLE, kernel density estimation, fractional OU estimation, and CIR parameter fitting
- **SIMD-optimized** - fractional Gaussian noise, fractional Brownian motion, and all probability distributions use wide SIMD for fast sample generation
- **Parallel sampling** - `sample_par(m)` generates `m` independent paths in parallel via rayon
- **Generic precision** - most models support both `f32` and `f64`
- **Bindings** - full stochastic model coverage with numpy integration; all models return numpy arrays

## Installation

### Rust

```toml
[dependencies]
stochastic-rs = "1.0.0"
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

CUDA build details (Windows/Linux commands) are documented in `src/stochastic/cuda/CUDA_BUILD.md`.

### CUDA fallback (if auto-build fails)

If `cargo build --features cuda` fails (for example: `nvcc fatal : Cannot find compiler 'cl.exe'`), use prebuilt CUDA FGN binaries.

1. Download the platform file from GitHub Releases:  
   `https://github.com/dancixx/stochastic-rs/releases`
2. Place it at:
   - Windows: `src/stochastic/cuda/fgn_windows/fgn.dll`
   - Linux: `src/stochastic/cuda/fgn_linux/libfgn.so`
3. Set runtime path explicitly:

```powershell
$env:STOCHASTIC_RS_CUDA_FGN_LIB_PATH='src/stochastic/cuda/fgn_windows/fgn.dll'
```

```bash
export STOCHASTIC_RS_CUDA_FGN_LIB_PATH=src/stochastic/cuda/fgn_linux/libfgn.so
```

### FGN CPU vs CUDA (`sample`, `sample_par`, `sample_cuda`)

Measured with Criterion in `--release` using:

```bash
$env:STOCHASTIC_RS_CUDA_FGN_LIB_PATH='src/stochastic/cuda/fgn_windows/fgn.dll'
cargo bench --bench fgn_cuda --features cuda -- --noplot
```

Environment:
- GPU: NVIDIA GeForce RTX 4070 SUPER
- Rust: `rustc 1.93.1`
- CUDA library: `src/stochastic/cuda/fgn_windows/fgn.dll` (fatbin `sm_75+`)

Note: one-time CUDA init is excluded via warmup (`sample_cuda(...)` called once before each benchmark case).

Single path (`sample` vs `sample_cuda(1)`, `f32`, H=0.7):

| n | CPU `sample` | CUDA `sample_cuda(1)` | CUDA speedup (CPU/CUDA) |
|---:|---:|---:|---:|
| 1,024 | 10.112 us | 62.070 us | 0.16x |
| 4,096 | 40.901 us | 49.040 us | 0.83x |
| 16,384 | 184.060 us | 59.592 us | 3.09x |
| 65,536 | 1.0282 ms | 121.160 us | 8.49x |

Batch (`sample_par(m)` vs `sample_cuda(m)`, `f32`, H=0.7):

| n, m | CPU `sample_par(m)` | CUDA `sample_cuda(m)` | CUDA speedup (CPU/CUDA) |
|---|---:|---:|---:|
| 4,096, 32 | 148.840 us | 154.080 us | 0.97x |
| 4,096, 128 | 364.690 us | 1.1255 ms | 0.32x |
| 4,096, 512 | 1.7975 ms | 4.3293 ms | 0.42x |
| 16,384, 128 | 1.7029 ms | 4.5458 ms | 0.37x |
| 16,384, 512 | 5.5850 ms | 17.2110 ms | 0.32x |

Interpretation:
- CUDA wins for large single-path generation (from roughly `n >= 16k` in this setup).
- For the tested batch sizes, CPU `sample_par` is faster than current CUDA path.

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
