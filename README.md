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

Requires [maturin](https://www.maturin.rs/):

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

## Contributing

Contributions are welcome — bug reports, feature suggestions, or PRs. Open an issue or start a discussion on GitHub.

## License

MIT — see [LICENSE](LICENSE).
