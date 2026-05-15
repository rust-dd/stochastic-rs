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

use ndarray::Array1;

pub mod coefficients;
mod engine;

pub use coefficients::convolution_coefficients;
pub use coefficients::fourier_coefficients_dx;
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

/// Default cutting frequencies for the Dirichlet/Cesàro **FE** spot-volatility
/// estimator (MATLAB FSDA `FE_spot_vol` / `FE_spot_vol_FFT` convention):
/// `N = floor((n − 1) / 2)`, `M = floor(√n · log(n))`.
///
/// The FE convention uses a wider Fejér window than the FM convention
/// (`M ≈ √N`) — pair this with [`FMVol::spot_variance_fe`] for parity with
/// the MATLAB output.
pub fn default_cutting_freq_fe(n: usize) -> (usize, usize) {
  let big_n = (n - 1) / 2;
  let big_m = ((n as f64).sqrt() * (n as f64).ln()).floor() as usize;
  (big_n, big_m)
}

/// Compute the MSE-optimal cutting frequency *N* for the Fourier estimator
/// of integrated variance in the presence of microstructure noise.
///
/// Implements the algorithm from:
/// - Mancino, M. E., Recchioni, M. C. & Sanfelici, S. (2017), §3.4.
///
/// # Algorithm
///
/// 1. **Sparse sampling**: skip observations until lag-1 autocorrelation
///    of increments falls below the 95 % Bartlett bound (±1.96/√n).
///    This removes noise-induced serial dependence.
/// 2. Estimate integrated variance (RV) and quarticity (Q) from sparse data.
/// 3. Estimate noise moments from the *original* (non-sparse) increments.
/// 4. Evaluate the MSE of the Fourier estimator for each candidate *N*
///    and return the minimiser.
///
/// # Returns
///
/// `OptimalCuttingResult { n_opt, noise_variance, mse_curve }`.
pub fn optimal_cutting_frequency(prices: &[f64], times: &[f64]) -> OptimalCuttingResult {
  assert_eq!(
    prices.len(),
    times.len(),
    "prices and times must have the same length"
  );
  let n_obs = prices.len();
  assert!(n_obs >= 8, "need at least 8 observations");
  let n = n_obs - 1; // number of increments

  let period = times[n_obs - 1] - times[0];
  assert!(period > 0.0, "time period must be positive");

  // sparse sampling
  // Compute increments of prices
  let increments = Array1::from_vec((0..n).map(|i| prices[i + 1] - prices[i]).collect());

  // Find sparse step so that lag-1 autocorrelation of increments is
  // below the Bartlett 95 % bound.
  let mut step = 1usize;
  loop {
    let sparse_inc = if step == 1 {
      increments.clone()
    } else {
      let sp: Vec<f64> = (0..n_obs).step_by(step).map(|i| prices[i]).collect();
      Array1::from_vec((0..sp.len() - 1).map(|i| sp[i + 1] - sp[i]).collect())
    };

    if sparse_inc.len() < 4 {
      break;
    }

    let acf1 = lag1_autocorrelation(&sparse_inc);
    let bound = 1.96 / (sparse_inc.len() as f64).sqrt();

    if acf1.abs() < bound || step >= n / 4 {
      break;
    }
    step += 1;
  }

  // Sparse prices and their increments
  let sparse_prices: Vec<f64> = (0..n_obs).step_by(step).map(|i| prices[i]).collect();
  let sparse_inc = Array1::from_vec(
    (0..sparse_prices.len() - 1)
      .map(|i| sparse_prices[i + 1] - sparse_prices[i])
      .collect(),
  );
  let n_sparse = sparse_inc.len();

  // RV and quarticity from sparse data
  let rv = sparse_inc.mapv(|r| r * r).sum();
  let sum4 = sparse_inc.mapv(|r| r.powi(4)).sum();
  let q = sum4 * n_sparse as f64 / (3.0 * period);

  // Noise moments from original data
  let sum_r2 = increments.mapv(|r| r * r).sum();
  let sum_r4 = increments.mapv(|r| r.powi(4)).sum();

  let e2 = sum_r2 / n as f64 - rv / n as f64;
  let e4 = sum_r4 / n as f64 - 6.0 * e2 * rv / n as f64;

  let eeta2 = e2 / 2.0;
  let eeta4 = e4 / 2.0 - 3.0 * e2 * e2 / 4.0;

  let alpha = e2 * e2;
  let beta = 4.0 * eeta4;
  let gamma = 8.0 * eeta2 * rv + alpha / 2.0 - 2.0 * eeta4;

  let noise_variance = eeta2.max(0.0);

  // MSE for each candidate N
  let n_max = n / 2;
  let h = period / n as f64;

  let mut mse_curve = Array1::<f64>::zeros(n_max);
  let mut best_mse = f64::INFINITY;
  let mut n_opt = 1usize;

  for k in 1..=n_max {
    let trunc = (n / 2).min(k);
    let rd = rescaled_dirichlet_kernel(trunc, h, period);
    let rd2 = rd * rd;

    let alpha_fe = alpha * (1.0 + rd2 - 2.0 * rd);
    let beta_fe = beta * (1.0 + rd2 - 2.0 * rd);
    let gamma_fe = gamma
      + 4.0 * (eeta4 + (e2 / 2.0).powi(2)) * (2.0 * rd - rd2)
      + 8.0 * std::f64::consts::PI * q / (2 * trunc + 1) as f64;

    let mse = 2.0 * q * h + beta_fe * n as f64 + alpha_fe * (n as f64).powi(2) + gamma_fe;

    mse_curve[k - 1] = mse;

    if mse < best_mse {
      best_mse = mse;
      n_opt = k;
    }
  }

  n_opt = n_opt.min(n_max).max(1);

  OptimalCuttingResult {
    n_opt,
    noise_variance,
    mse_curve,
  }
}

/// Result of [`optimal_cutting_frequency`].
pub struct OptimalCuttingResult {
  /// Optimal primary cutting frequency *N*.
  pub n_opt: usize,
  /// Estimated noise variance σ²_η.
  pub noise_variance: f64,
  /// MSE values for N = 1, …, n/2. The minimum is at index `n_opt - 1`.
  pub mse_curve: Array1<f64>,
}

impl OptimalCuttingResult {
  /// Derive the secondary cutting frequencies M and L from `n_opt`.
  ///
  /// Uses the noisy-data rule: M = floor(0.3 √N), L = floor(√M).
  pub fn cutting_freqs(&self) -> (usize, usize, usize) {
    let m = (0.3 * (self.n_opt as f64).sqrt()) as usize;
    let l = (m as f64).sqrt() as usize;
    (self.n_opt, m.max(1), l.max(1))
  }
}

/// Lag-1 sample autocorrelation of an ndarray.
fn lag1_autocorrelation(x: &Array1<f64>) -> f64 {
  let n = x.len();
  if n < 2 {
    return 0.0;
  }
  let mean = x.sum() / n as f64;
  let centered = x.mapv(|v| v - mean);
  let cov0 = centered.mapv(|v| v * v).sum();
  if cov0.abs() < 1e-30 {
    return 0.0;
  }
  let cov1: f64 = (1..n).map(|i| centered[i] * centered[i - 1]).sum();
  cov1 / cov0
}

/// Rescaled Dirichlet kernel: D_N(t) = (1/(2N+1)) Σ_{s=-N}^{N} e^{i s 2π t / T}.
///
/// For real evaluation this simplifies to (1 + 2 Σ_{s=1}^{N} cos(s·2π·h/T)) / (2N+1)
/// where h is the grid spacing.
fn rescaled_dirichlet_kernel(big_n: usize, h: f64, period: f64) -> f64 {
  let omega = std::f64::consts::TAU / period;
  let mut sum = 1.0;
  for s in 1..=big_n {
    sum += 2.0 * (s as f64 * omega * h).cos();
  }
  sum / (2 * big_n + 1) as f64
}
