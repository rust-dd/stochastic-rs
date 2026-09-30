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
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stats::hurst::HurstEstimator;
use stochastic_rs_stats::hurst::rs::RescaledRange;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::traits::ProcessExt;

// Fractional Gaussian noise with H = 0.3, then read H back from the path.
let path = Fgn::<f64, _>::new(0.3, 4096, Some(1.0), Deterministic::new(7)).sample();
let estimator = RescaledRange {
    take_differences: false,
    ..RescaledRange::default()
};
let est = estimator.estimate(path.view()).expect("a path long enough to estimate");
assert!((est.hurst - 0.3).abs() < 0.1);
```

The Fukasawa and Whittle estimators in `hurst::whittle` take price data
(`estimate_from_prices(closes)`) and estimate the roughness of volatility,
not the Hurst exponent of the path itself.

## Part of stochastic-rs

This crate is one of the sub-crates of
[**stochastic-rs**](https://github.com/rust-dd/stochastic-rs). Most users
should depend on the umbrella crate, which re-exports everything:

```toml
[dependencies]
stochastic-rs = "3.0.0-rc.4"
```

Depend on `stochastic-rs-stats` directly only when you want this slice and nothing else.

- Documentation: [stochastic.rust-dd.com](https://stochastic.rust-dd.com)
- Tutorials: [stochastic.rust-dd.com/docs/tutorials](https://stochastic.rust-dd.com/docs/tutorials)
- API reference: [docs.rs/stochastic-rs-stats](https://docs.rs/stochastic-rs-stats)

## License

MIT
