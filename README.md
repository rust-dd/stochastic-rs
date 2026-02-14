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
| Normal | f32 | 32 | 2.84 | 8.43 | 2.97x |
| Normal | f32 | 64 | 2.93 | 8.43 | 2.88x |
| Normal | f64 | 32 | 2.80 | 9.81 | 3.50x |
| Normal | f64 | 64 | 2.83 | 9.81 | 3.47x |
| Exp | f32 | 32 | 2.58 | 11.75 | 4.55x |
| Exp | f32 | 64 | 2.50 | 11.75 | 4.70x |
| Exp | f64 | 32 | 2.59 | 11.16 | 4.31x |
| Exp | f64 | 64 | 2.56 | 11.16 | 4.36x |
| LogNormal | f32 | - | 4.31 | 7.64 | 1.77x |
| LogNormal | f64 | - | 5.46 | 12.78 | 2.34x |
| Cauchy | f32 | - | 2.29 | 9.80 | 4.28x |
| Cauchy | f64 | - | 6.09 | 10.52 | 1.73x |
| Gamma | f32 | - | 6.08 | 12.14 | 2.00x |
| Gamma | f64 | - | 6.32 | 14.88 | 2.35x |
| Weibull | f32 | - | 5.00 | 7.34 | 1.47x |
| Weibull | f64 | - | 10.26 | 15.08 | 1.47x |
| Beta | f32 | - | 12.25 | 36.37 | 2.97x |
| Beta | f64 | - | 12.78 | 46.33 | 3.63x |
| ChiSquared | f32 | - | 5.99 | 12.28 | 2.05x |
| ChiSquared | f64 | - | 6.17 | 14.81 | 2.40x |
| StudentT | f32 | - | 9.09 | 19.67 | 2.16x |
| StudentT | f64 | - | 9.37 | 22.70 | 2.42x |
| Poisson | u32 | - | 21.42 | 40.37 | 1.88x |
| Pareto | f32 | - | 2.49 | 5.25 | 2.11x |
| Pareto | f64 | - | 4.81 | 10.95 | 2.28x |
| Uniform | f32 | - | 3.10 | 3.04 | 0.98x |
| Uniform | f64 | - | 5.65 | 5.62 | 1.01x |

### 100K samples (large dataset)

| Distribution | Type | N | stochastic-rs (µs) | rand_distr (µs) | Speedup |
|---|---|---|---:|---:|---:|
| Normal | f32 | 32 | 284 | 829 | 2.92x |
| Normal | f32 | 64 | 290 | 829 | 2.86x |
| Normal | f64 | 32 | 279 | 981 | 3.51x |
| Normal | f64 | 64 | 281 | 981 | 3.49x |
| Exp | f32 | 32 | 256 | 923 | 3.60x |
| Exp | f32 | 64 | 249 | 923 | 3.71x |
| Exp | f64 | 32 | 260 | 935 | 3.60x |
| Exp | f64 | 64 | 254 | 935 | 3.68x |
| LogNormal | f32 | - | 431 | 754 | 1.75x |
| LogNormal | f64 | - | 543 | 1293 | 2.38x |
| Cauchy | f32 | - | 229 | 978 | 4.27x |
| Cauchy | f64 | - | 593 | 1042 | 1.76x |
| Gamma | f32 | - | 608 | 1214 | 2.00x |
| Gamma | f64 | - | 631 | 1480 | 2.35x |
| Weibull | f32 | - | 501 | 733 | 1.46x |
| Weibull | f64 | - | 1027 | 1509 | 1.47x |
| Beta | f32 | - | 1223 | 3641 | 2.98x |
| Beta | f64 | - | 1280 | 4625 | 3.61x |
| ChiSquared | f32 | - | 596 | 1224 | 2.05x |
| ChiSquared | f64 | - | 617 | 1485 | 2.41x |
| StudentT | f32 | - | 906 | 1967 | 2.17x |
| StudentT | f64 | - | 936 | 2273 | 2.43x |
| Poisson | u32 | - | 2132 | 4036 | 1.89x |
| Pareto | f32 | - | 249 | 524 | 2.10x |
| Pareto | f64 | - | 481 | 1095 | 2.28x |
| Uniform | f32 | - | 310 | 303 | 0.98x |
| Uniform | f64 | - | 564 | 562 | 1.00x |

## Contributing

Contributions are welcome — bug reports, feature suggestions, or PRs. Open an issue or start a discussion on GitHub.

## License

MIT — see [LICENSE](LICENSE).
