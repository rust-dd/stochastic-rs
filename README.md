![Build Workflow](https://github.com/dancixx/stochastic-rs/actions/workflows/rust.yml/badge.svg)
[![Crates.io](https://img.shields.io/crates/v/stochastic-rs?style=flat-square)](https://crates.io/crates/stochastic-rs)
![License](https://img.shields.io/crates/l/stochastic-rs?style=flat-square)
[![codecov](https://codecov.io/gh/dancixx/stochastic-rs/graph/badge.svg?token=SCSp3z7BQJ)](https://codecov.io/gh/dancixx/stochastic-rs)

# stochastic-rs

A high-performance Rust library for simulating stochastic processes, with first-class bindings. Built for quantitative finance, statistical modeling and synthetic data generation.

## Features

- **85+ stochastic processes** — 31 diffusions (OU, CIR, GBM, CEV, CKLS, Aït-Sahalia, Pearson, Jacobi, regime-switching, …), 15 jump processes (Merton, Kou, CGMY, bilateral gamma, …), 9 stochastic-volatility models (Heston, SABR, Bergomi, rough Bergomi, HKDE, …), 13 short-rate / HJM / BGM models, plus base processes (fBM, fGN, Poisson, Hawkes, Lévy, LFSM, …). Each carries a generic-precision `ProcessExt<T>` impl and CUDA / SIMD acceleration where applicable.
- **Pricing** — closed-form (BSM, Bachelier, Black76, Garman-Kohlhagen, Margrabe, Kirk, Geske compound, Stulz best-of-two, Bjerksund-Stensland, digital / gap / supershare, geometric basket, Levy moment-matching, cliquet / forward-start chain) · Fourier (Carr-Madan, Lewis, Gil-Pelaez) for Heston / Bates / Merton-jump / Kou / VG / CGMY / HKDE / double-Heston · Monte Carlo (basket, rainbow, cliquet with cap/floor and memory, autocallable phoenix / athena, spread) · finite difference (explicit / implicit / Crank-Nicolson, American) · Bermudan LSM · Heston SLV (Guyon–Labordère)
- **Fixed income** — yield-curve bootstrapping (deposit / FRA / future / swap), Nelson-Siegel / Svensson, multi-curve (OIS vs SOFR), discount-curve interpolation (linear / log-linear / cubic / monotone-convex) · vanilla / OIS / basis / cross-currency IRS · fixed-rate / floating-rate / inflation-linked / amortizing bonds · YTM / Macaulay / modified duration / convexity / Z-spread / OAS · cap / floor / collar / European & Bermudan swaptions with Hull-White, Black-Karasinski and G2++ tree engines · Jamshidian analytic European swaption · SABR / Shifted-SABR caplet calibration · CMS with Hagan linear-TSR
- **Calibration** — Heston (Cui analytic Jacobian + NMLE / PMLE / NMLE-CEKF seeds), SABR per-expiry caplet smile, Lévy (CGMY, VG, NIG, Merton-jump, Kou, bilateral gamma), Stochastic Volatility Jump (SVJ), rough Bergomi, double Heston, BSM (multi-maturity), HKDE, Hull-White swaption-grid via Levenberg-Marquardt
- **Risk** — VaR (Gaussian / historical / Monte Carlo), CVaR / expected shortfall, drawdown metrics, Sharpe / Sortino / Information-Ratio / Calmar (no hard-coded annualisation), instrument-level Greeks via finite differences, bucket DV01, scenario / shock / curve-shift stress framework
- **Credit** — Merton structural model (PD, equity / debt, distance-to-default, credit spread, implied recovery), reduced-form survival / hazard curves, CDS pricing (ISDA daily-grid, fair spread, risky PV01), hazard bootstrap from CDS par-spread term structure, JLT migration matrices with pure-Rust Padé-13 matrix exponential
- **Inflation** — zero-coupon and YoY inflation curves, CPI / RPI / HICP indices with linear-interpolated reference ratio, ZC and YoY inflation swaps with par-rate solver
- **Microstructure** — Almgren-Chriss optimal execution, Kyle (1985) strategic-trading equilibria, Bouchaud propagator with power-law / exponential / custom kernels, Roll / Corwin-Schultz spread estimators, full price-time priority order book
- **Statistics** — Hurst estimators, ADF / KPSS / Phillips-Perron / Leybourne-McCabe / ERS stationarity, Jarque-Bera / Shapiro-Francia / Anderson-Darling normality, periodogram / spectrum-search · realized variance / bipower / MinRV / MedRV / flat-top kernel (Bartlett / Parzen / Tukey-Hanning / Cubic / Quadratic-Spectral) with BNHLS bandwidth, semivariance, realized skew / kurtosis, HAR-RV, Jacod pre-averaging, TSRV, multi-scale RV, BNS jump test · Engle-Granger and Johansen cointegration, Granger causality, Gaussian-emission HMM with Baum-Welch, CUSUM and PELT changepoint · particle filter / UKF / random-walk Metropolis-Hastings · MLE engine for 1-D diffusions with 6 transition-density approximations (Euler, Ozaki, Shoji-Ozaki, Elerian, Kessler, Aït-Sahalia) and L-BFGS via argmin, plus dedicated Heston MLE / NMLE-CEKF
- **Factors & strategies** — PCA, two-pass Fama-MacBeth, Ledoit-Wolf shrinkage, cointegrated pairs trading (hedge ratio, spread, z-score, signal generator), forecast-momentum-volatility regime engine
- **Distributions** — `DistributionSampler<T>`-driven SIMD bulk sampling and `sample_matrix` for normal, log-normal, exponential (uniform / ziggurat), beta, gamma, chi-squared, Student-t, Poisson, alpha-stable, NIG, bilateral gamma, binomial, Cauchy, Pareto, Weibull
- **Copulas** — Clayton, Frank, Gumbel, Joe, Galambos, AMH, Gaussian, Student-t, Plackett, FGM bivariate; Gaussian, Student-t, vine multivariate; empirical, with correlation utilities
- **Advanced Monte Carlo** — variance reduction (antithetic, control variates, importance sampling, stratified), quasi-MC (Sobol, Halton), Multi-Level Monte Carlo, Longstaff-Schwartz LSM
- **Volatility surface** — implied-vol surface from market data, SVI / SSVI, arbitrage-free interpolation / extrapolation, smile and skew analytics
- **Calendar & day count** — ACT/360, ACT/365, 30/360, ACT/ACT · Following / Modified Following / Preceding · US, UK, TARGET, Tokyo holiday calendars · pluggable `CalendarExt` · `ScheduleBuilder` for coupon / payment dates
- **FX** — ISO 4217 currency definitions, FX quoting / cross-rate / triangulation, FX forward via covered interest parity (continuous and simple compounding)
- **Performance** — wide SIMD (`f64x4` / `f32x8`) for FGN, fBM, all distributions; `sample_par(m)` for `m` independent paths via rayon; CUDA backend for FGN; thread-local FFT scratch buffers
- **Generic precision** — all numerical code is generic over `T: FloatExt`, supporting both `f32` and `f64`
- **Python bindings** — full coverage of stochastic models with numpy integration; all sample paths return numpy arrays

## Installation

### Rust — umbrella crate (everything)

```toml
[dependencies]
stochastic-rs = "2.0.0-beta.2"
```

```rust
use stochastic_rs::prelude::*; // FloatExt, ProcessExt, ModelPricer, OptionType, ...
use stochastic_rs::stochastic::diffusion::gbm::GBM;
use stochastic_rs::quant::pricing::heston::HestonPricer;
```

### Rust — pick the sub-crates you need

The crate is split into a workspace; pull in only what you use to keep build
times and dependency surface minimal.

```toml
[dependencies]
stochastic-rs-distributions = "2.0.0-beta.2"  # SIMD distribution sampling
stochastic-rs-stochastic    = "2.0.0-beta.2"  # 140+ process types
stochastic-rs-copulas       = "2.0.0-beta.2"  # bivariate / multivariate copulas
stochastic-rs-stats         = "2.0.0-beta.2"  # estimators
stochastic-rs-quant         = "2.0.0-beta.2"  # pricing / calibration / vol surface
stochastic-rs-ai            = "2.0.0-beta.2"  # neural surrogates (candle)
stochastic-rs-viz           = "2.0.0-beta.2"  # plotly grid plotter
```

Topology:

```
stochastic-rs-core (simd_rng)
 └→ stochastic-rs-distributions (FloatExt, SimdFloatExt, distribution types)
     ├→ stochastic-rs-stochastic (ProcessExt + 140+ processes)
     ├→ stochastic-rs-copulas (BivariateExt, etc.)
     └→ stochastic-rs-stats (estimators)
         └→ stochastic-rs-quant (PricerExt, ModelPricer, calibration, vol surface)
             ├→ stochastic-rs-ai (HestonNn / OneFactorNn / RoughBergomiNn)
             └→ stochastic-rs-viz (GridPlotter)
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
