![Build Workflow](https://github.com/dancixx/stochastic-rs/actions/workflows/rust.yml/badge.svg)
[![Crates.io](https://img.shields.io/crates/v/stochastic-rs?style=flat-square)](https://crates.io/crates/stochastic-rs)
![License](https://img.shields.io/crates/l/stochastic-rs?style=flat-square)
[![codecov](https://codecov.io/gh/dancixx/stochastic-rs/graph/badge.svg?token=SCSp3z7BQJ)](https://codecov.io/gh/dancixx/stochastic-rs)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fdancixx%2Fstochastic-rs.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fdancixx%2Fstochastic-rs?ref=badge_shield)

# stochastic-rs

A high-performance Rust library for simulating stochastic processes, with first-class bindings. Built for quantitative finance, statistical modeling and synthetic data generation.

## Features

- **85+ stochastic models** — diffusions, jump processes, stochastic volatility, interest rate models, autoregressive models, noise generators, and probability distributions
- **Copulas** — bivariate, multivariate, and empirical copulas with correlation utilities
- **Quant toolbox** — option pricing, bond analytics, calibration, loss models, order book, and trading strategies
- **Statistics** — MLE, kernel density estimation, fractional OU estimation, and CIR parameter fitting
- **SIMD-optimized** — fractional Gaussian noise, fractional Brownian motion, and all probability distributions use wide SIMD for fast sample generation
- **Parallel sampling** — `sample_par(m)` generates `m` independent paths in parallel via rayon
- **Generic precision** — most models support both `f32` and `f64`
- **Bindings** — full stochastic model coverage with numpy integration; all models return numpy arrays

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
use stochastic_rs::traits::ProcessExt;

fn main() {
    // Fractional Brownian Motion
    let fbm = FBM::new(0.7, 1000, None);
    let path = fbm.sample();

    // Parallel batch sampling
    let paths = fbm.sample_par(1000);

    // Heston stochastic volatility
    let heston = Heston::new(0.05, 2.0, 0.04, 0.3, -0.7, 1000, Some(100.0), Some(0.04), None, None);
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

Distribution sampling performance: `stochastic-rs` SIMD vs `rand_distr`.
All distributions use an internal SIMD PRNG (xoshiro256++/xoshiro128++ on `wide` SIMD types) for maximum throughput.
For Normal and Exp, the const generic buffer size (N=32 / N=64) is also compared.
Measured with Criterion on Apple M-series, `--release`.

### 1K samples (small dataset)

| Distribution | Type | N | stochastic-rs (µs) | rand_distr (µs) | Speedup |
|---|---|---|---:|---:|---:|
| Normal | f32 | 32 | 1.98 | 8.30 | 4.19x |
| Normal | f32 | 64 | 2.09 | 8.30 | 3.97x |
| Normal | f64 | 32 | 2.02 | 9.72 | 4.81x |
| Normal | f64 | 64 | 2.14 | 9.72 | 4.54x |
| Exp | f32 | 32 | 1.80 | 9.23 | 5.13x |
| Exp | f32 | 64 | 1.79 | 9.23 | 5.16x |
| Exp | f64 | 32 | 1.87 | 9.26 | 4.95x |
| Exp | f64 | 64 | 1.85 | 9.26 | 5.01x |
| LogNormal | f32 | - | 2.90 | 7.68 | 2.65x |
| LogNormal | f64 | - | 4.57 | 12.91 | 2.83x |
| Cauchy | f32 | - | 2.31 | 9.98 | 4.32x |
| Cauchy | f64 | - | 6.25 | 10.44 | 1.67x |
| Gamma | f32 | - | 5.26 | 12.31 | 2.34x |
| Gamma | f64 | - | 5.60 | 14.94 | 2.67x |
| Weibull | f32 | - | 5.00 | 7.36 | 1.47x |
| Weibull | f64 | - | 10.25 | 15.10 | 1.47x |
| Beta | f32 | - | 10.64 | 36.43 | 3.42x |
| Beta | f64 | - | 11.32 | 46.46 | 4.11x |
| ChiSquared | f32 | - | 5.16 | 12.32 | 2.39x |
| ChiSquared | f64 | - | 5.49 | 14.79 | 2.69x |
| StudentT | f32 | - | 7.50 | 19.69 | 2.63x |
| StudentT | f64 | - | 7.83 | 22.58 | 2.88x |
| Poisson | u32 | - | 21.95 | 41.13 | 1.87x |
| Pareto | f32 | - | 2.51 | 5.28 | 2.10x |
| Pareto | f64 | - | 4.90 | 11.01 | 2.25x |
| Uniform | f32 | - | 3.08 | 3.05 | 0.99x |
| Uniform | f64 | - | 5.69 | 5.65 | 0.99x |

### 100K samples (large dataset)

| Distribution | Type | N | stochastic-rs (µs) | rand_distr (µs) | Speedup |
|---|---|---|---:|---:|---:|
| Normal | f32 | 32 | 196 | 830 | 4.23x |
| Normal | f32 | 64 | 209 | 830 | 3.97x |
| Normal | f64 | 32 | 201 | 973 | 4.84x |
| Normal | f64 | 64 | 211 | 973 | 4.61x |
| Exp | f32 | 32 | 180 | 934 | 5.19x |
| Exp | f32 | 64 | 180 | 934 | 5.19x |
| Exp | f64 | 32 | 188 | 924 | 4.91x |
| Exp | f64 | 64 | 185 | 924 | 4.99x |
| LogNormal | f32 | - | 291 | 763 | 2.62x |
| LogNormal | f64 | - | 468 | 1284 | 2.74x |
| Cauchy | f32 | - | 231 | 1010 | 4.37x |
| Cauchy | f64 | - | 593 | 1044 | 1.76x |
| Gamma | f32 | - | 525 | 1227 | 2.34x |
| Gamma | f64 | - | 560 | 1490 | 2.66x |
| Weibull | f32 | - | 502 | 733 | 1.46x |
| Weibull | f64 | - | 1025 | 1510 | 1.47x |
| Beta | f32 | - | 1062 | 3645 | 3.43x |
| Beta | f64 | - | 1129 | 4652 | 4.12x |
| ChiSquared | f32 | - | 513 | 1235 | 2.41x |
| ChiSquared | f64 | - | 545 | 1478 | 2.71x |
| StudentT | f32 | - | 744 | 1969 | 2.65x |
| StudentT | f64 | - | 784 | 2332 | 2.97x |
| Poisson | u32 | - | 2166 | 4235 | 1.96x |
| Pareto | f32 | - | 251 | 527 | 2.10x |
| Pareto | f64 | - | 485 | 1103 | 2.27x |
| Uniform | f32 | - | 307 | 306 | 1.00x |
| Uniform | f64 | - | 568 | 566 | 1.00x |

## Contributing

Contributions are welcome — bug reports, feature suggestions, or PRs. Open an issue or start a discussion on GitHub.

## License

MIT — see [LICENSE](LICENSE).
