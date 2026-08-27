[![Crates.io](https://img.shields.io/crates/v/stochastic-rs-stats?style=flat-square)](https://crates.io/crates/stochastic-rs-stats)
[![docs.rs](https://img.shields.io/docsrs/stochastic-rs-stats?style=flat-square)](https://docs.rs/stochastic-rs-stats)
![License](https://img.shields.io/crates/l/stochastic-rs-stats?style=flat-square)

# stochastic-rs-stats

**Statistical estimators for stochastic processes**

Estimation and testing for the processes simulated elsewhere in the
workspace. Every estimator is anchored to a published paper.

## What is in it

- **Hurst exponent** — Fukasawa, rescaled range, DFA, GPH, wavelet,
  Whittle, variogram and Higuchi fractal dimension.
- **Maximum likelihood** — 1-D diffusions with six transition-density
  approximations, plus QMLE, GMM for CIR, Heston MLE, particle MLE and a
  non-linear-marginal CEKF for Heston.
- **Realised measures** — realised variance, bipower variation, two-scale
  and pre-averaging estimators, realised kernels with BNHLS bandwidth, HAR.
- **Stationarity** — ADF, KPSS, Phillips-Perron, ERS-DFGLS,
  Leybourne-McCabe, Lo-MacKinlay, Andrews-Ploberger, CUSUM, RESET.
- **Normality** — Jarque-Bera, Anderson-Darling, Shapiro-Francia.
- **Econometrics** — cointegration, Granger causality, hidden Markov
  models, changepoint detection.
- **Filtering** — particle filter, unscented Kalman filter, MCMC.
- **Tails and spectra** — tail index estimation, periodogram-based spectral
  search.

## Usage

```rust
use ndarray::ArrayView1;
use stochastic_rs_stats::hurst::whittle;

let res = whittle::estimate_from_prices(ArrayView1::from(&closes));
```

## Part of stochastic-rs

This crate is one of the sub-crates of
[**stochastic-rs**](https://github.com/rust-dd/stochastic-rs). Most users
should depend on the umbrella crate, which re-exports everything:

```toml
[dependencies]
stochastic-rs = "3.0.0-beta.2"
```

Depend on `stochastic-rs-stats` directly only when you want this slice and nothing else.

- Documentation: [stochastic.rust-dd.com](https://stochastic.rust-dd.com)
- API reference: [docs.rs/stochastic-rs-stats](https://docs.rs/stochastic-rs-stats)

## License

MIT
