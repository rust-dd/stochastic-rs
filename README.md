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

## Roadmap to v2

> Turning stochastic-rs into a comprehensive **quantitative finance library** — a full QuantLib competitor in Rust.
>
> Items are ordered by dependency: each tier builds on the one above it. Independent modules (stats, factors) can be done in parallel at any time.

### Tier 0 — Foundations (no roadmap dependencies, builds on done modules)

- [x] **Cash flow engine** (`quant::cashflows`) ← *calendar ✓, curves ✓*
  - [x] Fixed-rate coupon, floating-rate coupon (IBOR, OIS), CMS coupon
  - [x] Leg abstraction — ordered sequence of cash flows with notional schedule
  - [x] Amortizing and accreting notionals
  - [x] Cash flow NPV, accrued interest
- [x] **Trinomial tree & lattice framework** (`quant::lattice`)
  - [x] General trinomial / binomial tree
  - [x] Hull-White, Black-Karasinski, G2++ tree engines

### Tier 1 — Instruments (depends on: cash flow engine)

- [x] **Interest rate swaps** (`quant::instruments`) ← *cashflows, curves ✓*
  - [x] Vanilla IRS (fixed vs floating)
  - [x] Overnight Indexed Swap (OIS)
  - [x] Basis swap and cross-currency basis swap
  - [x] NPV, fair rate, DV01 / BPV
- [x] **Fixed-income instruments** (`quant::instruments`) ← *cashflows, curves ✓*
  - [x] Fixed-rate bond — dirty/clean price, YTM, duration (Macaulay, modified), convexity
  - [x] Floating-rate bond / FRN
  - [x] Zero-coupon bond pricing from yield curve
  - [x] Amortizing bond, inflation-linked bond
  - [x] Z-spread, ASW spread, OAS

### Tier 2 — Advanced instruments & framework (depends on: Tier 1)

- [x] **Caps, floors & swaptions** (`quant::instruments::option`) ← *IRS, lattice*
  - [x] Cap / Floor / Collar pricing (Black, Bachelier, SABR)
  - [x] European and Bermudan swaptions
  - [x] Hull-White tree-based Bermudan engine (Black-Karasinski via `price_on_tree`)
  - [x] G2++ tree-based Bermudan engine (two-factor joint backward induction)
  - [x] Hull-White time-dependent $\theta(t)$ calibrated to initial yield curve (`CurveFittedHullWhiteTree`)
  - [x] Jamshidian analytic European swaption under Hull-White (ZBP strip)
  - [x] Hull-White calibration to swaption vol grids (`HullWhiteSwaptionCalibrator`)
  - [x] SABR calibration to per-expiry caplet smile (`SabrCapletCalibrator`)
  - [x] Shifted SABR for negative / low rates (`ShiftedSabrVolatility`)
  - [x] CMS caplet / floorlet with Hagan linear-TSR convexity adjustment
  - [x] Calendar-aware Bermudan API (snap `&[NaiveDate]` to tree levels via `BermudanSwaption::from_calendar`)
- [x] **Market data framework** (`quant::market`) ← *IRS, fixed-income (rate helpers)*
  - [x] Quote / Handle / Observable abstraction for reactive repricing (`SimpleQuote`, `DerivedQuote`, `CompositeQuote`, `Handle`, `RelinkableHandle`, `ObservableBase`)
  - [x] Named rate indices (SOFR, ESTR, SONIA, TONAR, Fed Funds, Euribor, USD Libor) with ARRC/ISDA conventions and observable `FixingHistory`
  - [x] Dated `ForwardRateAgreement` and `Deposit` instruments with NPV / par-rate under multi-curve discounting
  - [x] `RateHelper` trait + `DepositRateHelper` / `FraRateHelper` / `SwapRateHelper` / `FuturesRateHelper` feeding `build_curve` for quote-driven bootstrapping
- [x] **Credit models** (`quant::credit`) ← *fixed-income (CDS cash flows)*
  - [x] Merton structural model (analytical PD, equity, debt, distance-to-default, credit spread, implied recovery)
  - [x] Reduced-form / intensity-based models via `SurvivalCurve` / `HazardRateCurve` with piecewise-constant hazard interpolation
  - [x] CDS pricing (ISDA-style daily-grid integration) with protection, premium and accrual-on-default legs; fair spread and risky PV01
  - [x] Hazard rate bootstrapping from a CDS par-spread term structure (`bootstrap_hazard`)
  - [x] Default probability term structure via `SurvivalCurve::default_probability` and `conditional_default_probability`
  - [x] Credit migration matrices — discrete `TransitionMatrix` plus continuous-time `GeneratorMatrix` with a pure-Rust scaling-and-squaring Padé-13 matrix exponential, JLT embedding and Israel-Rosenthal-Wei generator projection
- [x] **Risk metrics** (`quant::risk`) ← *instruments (Greeks, bucket DV01)*
  - [x] Value at Risk — parametric Gaussian, historical-simulation and Monte-Carlo (`gaussian_var`, `historical_var`, `monte_carlo_var`, dispatcher `value_at_risk` + `VarMethod`)
  - [x] CVaR / Expected Shortfall (`gaussian_es`, `historical_es`, `monte_carlo_es`, dispatcher `expected_shortfall`)
  - [x] Stress testing and scenario analysis framework — `Scenario`, `Shock` (additive/multiplicative/level), `CurveShift` (parallel/twist/key-rate/at-pillars), `StressTest` engine
  - [x] Drawdown metrics (`running_drawdown`, `max_drawdown`, `max_drawdown_duration`, `DrawdownStats`), Sharpe, Sortino, Information Ratio, Calmar (all with explicit annualisation factor — no hard-coded `252`/`365`)
  - [x] Instrument-level Greeks — `central_difference`, `forward_difference`, `second_difference`, `bucket_dv01` with per-pillar PnL attribution via `CurveShift::KeyRate`

### Tier 3 — Complex products (depends on: Tier 2)

- [x] **Exotic derivatives** (`quant::pricing`) ← *lattice, pricing ✓, optionally swaptions*
  - [x] Bermudan options — Longstaff-Schwartz LSM (`BermudanLsmPricer`, requires `openblas`); PDE backstop via existing `finite_difference` American engine restricted to exercise dates
  - [x] Basket and rainbow options — geometric closed-form, Levy moment-matching, MC; Stulz best-of-two and `n`-asset MC rainbow
  - [x] Cliquet / ratchet options — closed-form forward-start chain plus MC with global cap/floor and memory
  - [x] Auto-callable structures — phoenix and athena (memory) under continuous, discrete, and at-maturity knock-in
  - [x] Digital / binary options — cash-or-nothing, asset-or-nothing, gap, supershare with closed-form Greeks
  - [x] Chooser, compound, spread options — simple and complex chooser, Geske compound (CoC/PoC/CoP/PoP), Margrabe and MC spread (Kirk already in `kirk.rs`)
- [x] **Inflation** (`quant::inflation`) ← *fixed-income, cashflows*
  - [x] Zero-coupon and year-on-year inflation term structures (`ZeroCouponInflationCurve`, `YoyInflationCurve` with the shared `InflationCurve` trait)
  - [x] CPI / RPI / HICP index objects (`PriceIndex`, `FixingHistory` with linear-interpolated reference ratio)
  - [x] Inflation-linked swaps (`ZeroCouponInflationSwap`, `YearOnYearInflationSwap` with par-rate solvers); inflation-linked bond already lived in `quant::instruments::bond::inflation_linked`

### Independent — Stats & quant modules (no tier dependencies)

- [x] **Realized volatility & microstructure noise** (`stats::realized`)
  - [x] Realized variance, bipower variation, MinRV, MedRV, tripower quarticity, flat-top realized kernel (Bartlett/Parzen/Tukey-Hanning/Tukey-Hanning2/Cubic/Quadratic-Spectral) with BNHLS bandwidth heuristic
  - [x] Realized semivariance (downside / upside), realized skewness and kurtosis (Amaya et al. 2015)
  - [x] HAR-RV (Corsi 2009) with OLS fit and one-step-ahead forecasts
  - [x] Noise-robust estimators — Jacod et al. pre-averaging, Zhang-Mykland-Aït-Sahalia TSRV, Zhang multi-scale RV
  - [x] Barndorff-Nielsen / Shephard ratio jump test (Huang–Tauchen finite-sample variant)
- [x] **Econometrics** (`stats::econometrics`)
  - [x] Cointegration — Engle-Granger two-step (with Phillips-Ouliaris critical values) and Johansen trace test on the VECM eigenvalue problem
  - [x] Granger causality F-test on nested VAR(p) regressions with regularised-incomplete-beta p-values
  - [x] Gaussian-emission Hidden Markov Model with Baum-Welch training, log-space scaled forward-backward and Viterbi decoding
  - [x] Changepoint detection — Page (1954) CUSUM control chart and Killick-Fearnhead-Eckley (2012) PELT for the squared-error mean-shift cost
- [x] **Bayesian inference & filtering** (`stats::filtering`)
  - [x] Bootstrap / SIR particle filter with multinomial / systematic / stratified resampling and ESS-triggered adaptation
  - [x] Unscented Kalman Filter — single-step predict + update with $2n+1$ sigma points and tunable $(\alpha, \beta, \kappa)$
  - [x] Random-Walk Metropolis-Hastings sampler with Gaussian proposal, burn-in and acceptance-rate diagnostics
- [x] **Factor models & statistical arbitrage** (`quant::factors`)
  - [x] PCA factor decomposition via SVD with explained-variance ratios, loadings and time-series scores
  - [x] Two-pass Fama-MacBeth cross-sectional regression with t-statistics and per-period premia series
  - [x] Ledoit-Wolf (2004) covariance shrinkage to scaled-identity target with data-driven optimal intensity
  - [x] Cointegrated pairs-trading framework — hedge-ratio OLS, spread, z-score, entry/exit signal generator
- [x] **Market microstructure** (`quant::microstructure`) ← *order_book ✓, hawkes ✓*
  - [x] Almgren-Chriss closed-form optimal execution with mean-variance frontier, buy/sell support, expected cost and variance decomposition
  - [x] Single-period and multi-period Kyle (1985) strategic-trading equilibria via backward $\alpha$-recursion
  - [x] Bouchaud propagator transient-impact model with power-law / exponential / custom kernels (path and current-tick impact)
  - [x] Roll (1984) implicit spread, realised effective spread, Corwin-Schultz (2012) high-low spread estimator

### Quality improvements (pre-v2 hardening)

> These should be addressed before or alongside Tier 0 work to ensure new modules build on a solid foundation.

- [x] **Stability: replace `.unwrap()` in core pricing** — `bsm.rs` (30 → 0) and `sabr.rs` (4 `tau` calls → 0) now route through `tau_required()` / explicit `expect` with descriptive panic messages. `heston.rs` already clean.
- [x] **Missing derives on public types** — added `Debug` + `Clone` to `OrderBook`, `BSMPricer`, `BSMPricerBuilder`, `FiniteDifferenceMethod` and ~35 pricing/calibration structs. Foundational types (`FGN`, `CGNS`, `Fn1D`, `SimdRng`) hold non-`Debug`/`Clone` external types (`FftHandler`, etc.) and need manual impls — tracked separately.
- [ ] **`Vec<Vec<f64>>` → `Array2<T>`** — `portfolio/{data,momentum,optimizers}.rs` (~1800 lines), `calibration/{bsm,rbergomi}.rs` (Jacobian cache, Cholesky), `pricing/{dupire,breeden_litzenberger}.rs`. Coordinated refactor needed.
- [x] **Mixed linalg libraries** — `calibration/*` submodules keep `nalgebra::{DMatrix, DVector}` at the Levenberg-Marquardt boundary (the `levenberg_marquardt` crate is nalgebra-based and there is no equivalent ndarray-native LM solver). The rest of the codebase stays on `ndarray`; the calibration boundary is the only `nalgebra` surface.
- [x] **Hardcoded day count constants** — `sabr_smile.rs:296` magic `1.0/365.0` equality check replaced with adaptive iteration count (`tau < 7/365`). `variance_swap.rs` `/252.0` is in test fixtures only; `fmvol_regime.rs` no longer contains hardcoded constants. Test-fixture `tau` literals in `pricing/sabr.rs` and `pricing/cgmysv` remain.
- [x] **Global `#![allow(dead_code)]`** — removed from `lib.rs`. Surfaced 16 unused items, all promoted to `pub` (helpers in `isonormal.rs`, `sample_with_seed`/`sample_cpu_with_seed` on noise generators, `Xoshiro256PP4`/`Xoshiro128PP8`+`new_from_rng`, `derive_seed`, `EmpiricalCopula2D::sample`, `RainbowPayoff::evaluate`, `HestonMalliavinGreeks::simulate`, `CEV::malliavin`, `volatility::sabr::malliavin_of_vol`).
- [ ] **Inline unit tests** — integration tests exist in `tests/{calendar_fx,curves,cashflows,instruments,market}_test.rs`, but `src/quant/{calendar,bonds,curves,fx}/*` and `src/stochastic/interest/*` lack `#[cfg(test)]` modules.
- [x] **Module documentation** — all listed files (`curves/*`, `bonds/*`, `calendar/holiday.rs`, `pricing/dupire.rs`, `pricing/breeden_litzenberger.rs`) now carry `//!` doc headers with formulas.
- [x] **Re-export consistency** — `bonds.rs` now re-exports `CIR`, `HullWhite`, `Vasicek`. `stochastic.rs` keeps trait-only re-exports by design (large submodule count, name collisions across `interest::CIR` and `bonds::CIR`).
- [ ] **Naming conventions** — `RBergomiCalibrationConfig` and `HestonNMLECEKFConfig` mix `Config`/`Params`. `interest/` filename prefixes (`mod_duffie_kan`, `fvasicek`, `hull_white_2f`) inconsistent.

### Done

- [x] **Advanced Monte Carlo** (`stochastic::mc`)
  - [x] Variance reduction — antithetic variates, control variates, importance sampling, stratified sampling
  - [x] Quasi-Monte Carlo sequences — Sobol, Halton
  - [x] Multi-Level Monte Carlo (MLMC)
  - [x] Longstaff-Schwartz (LSM) for American option pricing
- [x] **Volatility surface** (`quant::vol_surface`)
  - [x] Implied volatility surface construction from market data
  - [x] SVI parameterization (Gatheral) and SSVI
  - [x] Arbitrage-free interpolation and extrapolation
  - [x] Smile and skew analytics
- [x] **Calendar & day count** (`quant::calendar`)
  - [x] Day count conventions — ACT/360, ACT/365, 30/360, ACT/ACT
  - [x] Business day adjustment rules — Following, Modified Following, Preceding
  - [x] Holiday calendars — US, UK, TARGET, Tokyo
  - [x] Schedule generation for coupon and payment dates
- [x] **FX & currencies** (`quant::fx`)
  - [x] ISO 4217 currency definitions and metadata
  - [x] FX quoting and cross-rate conventions
  - [x] FX forward pricing
- [x] **Yield curve construction** (`quant::curves`)
  - [x] Bootstrapping from deposit, FRA, futures, and swap rates
  - [x] Nelson-Siegel / Svensson parameterization
  - [x] Discount factor and forward rate extraction
  - [x] Multi-curve framework (OIS vs SOFR discounting)
  - [x] Interpolation — monotone convex, log-linear, cubic spline
- [x] **Stochastic local volatility** (`quant::pricing`)
  - [x] Heston SLV model — Guyon–Labordère particle calibration of leverage function
  - [x] Combined stochastic + Dupire local vol with mixing factor η

## Contributing

Contributions are welcome - bug reports, feature suggestions, or PRs. Open an issue or start a discussion on GitHub.

## License

MIT - see [LICENSE](https://github.com/dancixx/stochastic-rs/blob/main/LICENSE).
