//! # Fourier-Malliavin Volatility Estimation
//!
//! $$
//! \widehat\sigma^2_{n,N,M}(\tau)=\sum_{|k|\le M}\!\Bigl(1-\frac{|k|}{M+1}\Bigr)c_k(\sigma_{n,N})\,e^{i\frac{2\pi}{T}k\tau}
//! $$
//!
//! Non-parametric estimation of volatility, covariance, leverage,
//! volatility-of-volatility and quarticity from observed price paths
//! using the Fourier-Malliavin method of Malliavin & Mancino.
//!
//! The primary API is the [`FMVol`] engine struct which pre-computes
//! the Fourier coefficients of price increments once and exposes all
//! estimators as cheap method calls.
//!
//! # Example
//! ```ignore
//! use stochastic_rs::quant::fourier_malliavin::FMVol;
//!
//! let engine = FMVol::new(&log_prices, &times, 1.0_f64);
//!
//! // Integrated quantities
//! let iv = engine.integrated_variance();
//! let il = engine.integrated_leverage(None);
//!
//! // Spot quantities at evaluation times
//! let tau: Vec<f64> = (0..51).map(|i| i as f64 / 50.0).collect();
//! let spot_var = engine.spot_variance(&tau, None);
//! ```
//!
//! # References
//!
//! - Sanfelici, S. & Toscano, G. (2024). *The Fourier-Malliavin Volatility
//!   (FMVol) MATLAB® library*. arXiv preprint [arXiv:2402.00172](https://arxiv.org/abs/2402.00172).
//! - Malliavin, P. & Mancino, M. E. (2002). Fourier series method for
//!   measurement of multivariate volatilities. *Finance and Stochastics*, 6(1), 49–61.
//! - Malliavin, P. & Mancino, M. E. (2009). A Fourier transform method for
//!   nonparametric estimation of multivariate volatility. *The Annals of Statistics*, 37(4), 1983–2010.
//! - Mancino, M. E., Recchioni, M. C. & Sanfelici, S. (2017). *Fourier-Malliavin
//!   Volatility Estimation: Theory and Practice*. Springer Briefs in Quantitative Finance.
//!
//! # MATLAB reference implementation
//!
//! FSDA Toolbox: <https://github.com/UniprJRC/FSDA>

pub mod coefficients;
mod engine;

pub use coefficients::{convolution_coefficients, fourier_coefficients_dx};
pub use engine::FMVol;

/// Default cutting frequencies for variance/covariance: N = floor(n/2), M = floor(N^0.5).
pub fn default_cutting_freq_vol(n: usize) -> (usize, usize) {
  let big_n = n / 2;
  let big_m = (big_n as f64).sqrt() as usize;
  (big_n, big_m)
}

/// Default cutting frequencies for leverage: N = floor(n/2), M = floor(N^0.5), L = floor(N^0.25).
pub fn default_cutting_freq_lev(n: usize) -> (usize, usize, usize) {
  let big_n = n / 2;
  let big_m = (big_n as f64).sqrt() as usize;
  let big_l = (big_n as f64).powf(0.25) as usize;
  (big_n, big_m, big_l)
}

/// Default cutting frequencies for vol-of-vol: N = floor(n/2), M = floor(N^0.4), L = floor(N^0.2).
pub fn default_cutting_freq_volvol(n: usize) -> (usize, usize, usize) {
  let big_n = n / 2;
  let big_m = (big_n as f64).powf(0.4) as usize;
  let big_l = (big_n as f64).powf(0.2) as usize;
  (big_n, big_m, big_l)
}

/// Default cutting frequencies with microstructure noise.
pub fn default_cutting_freq_noisy(n: usize) -> (usize, usize, usize) {
  let big_n = (5.0 * (n as f64).sqrt()) as usize;
  let big_m = (0.3 * (big_n as f64).sqrt()) as usize;
  let big_l = (big_m as f64).sqrt() as usize;
  (big_n, big_m, big_l)
}
