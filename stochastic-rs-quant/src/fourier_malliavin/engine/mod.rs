//! Struct-based Fourier-Malliavin volatility estimation engine.
//!
//! References:
//!   - Sanfelici & Toscano (2024), arXiv:2402.00172 — FMVol MATLAB library paper.
//!   - Malliavin & Mancino (2002, 2009) — original method.
//!   - Mancino & Recchioni (2015) — bias/variance analysis of second-order estimators.
//!   - Toscano, Livieri, Mancino, Marmi (2024), arXiv:2112.14529 — CLTs for the volvol
//!     Fourier estimator; eq.(3) defines the consistent estimator, eq.(4) the bias-corrected
//!     rate-n^{1/4} variant, eq.(51) gives the bias-correction constant K.
//!
//! ## Numerical validation
//!
//! All estimators are validated against Heston-Sqrt ground-truth at fixture parameters
//! (σ_v = 1.0, ρ = -0.5, V̄ = 0.4, T = 1.0, n = 23401 points):
//!
//! | Estimator                          | Ground-truth formula            | Test tolerance |
//! |------------------------------------|---------------------------------|----------------|
//! | `integrated_variance`              | ∫V_t dt (trapezoidal of v)      | rel_err < 15%  |
//! | `spot_variance`                    | V_τ                             | MAE < 0.25     |
//! | `integrated_leverage`              | σ_v · ρ · IV(T)                 | rel_err < 40%  |
//! | `integrated_volvol` (eq.3)         | σ_v² · IV(T)                    | factor-of-3    |
//! | `integrated_volvol_bias_corrected` | σ_v² · IV(T)                    | rel_err < 30%  |
//! | `spot_leverage` (mean)             | σ_v · ρ · mean(V_τ)             | rel_err < 30%  |
//! | `spot_volvol` (mean, eq.3)         | σ_v² · mean(V_τ)                | factor-of-3    |
//! | `spot_volvol_bias_corrected`       | σ_v² · mean(V_τ)                | rel_err < 40%  |
//!
//! The `integrated_volvol` and `spot_volvol` (eq.3) estimators have a known ~2× finite-sample
//! bias documented in Toscano-Livieri-Mancino-Marmi (2024) §3-§4. The bias-corrected
//! variants (eq.4) subtract `K · quarticity` where `K = M²/(3n)` for the uniform-sampling
//! default, reducing the bias to <10% on the Heston fixture.
//!
//! Tolerances reflect finite-sample variance of high-order moment estimators on a single
//! Heston path. Tests catch structural bugs (sign errors, missing factors of T or 2π)
//! while accommodating the expected estimator noise; the bias-corrected variants get
//! much tighter tolerances because the dominant bias term is removed.

use ndarray::Array1;
use num_complex::Complex;

use super::coefficients::fourier_coefficients_dx;
use super::coefficients::fourier_coefficients_dx_uniform;
use crate::traits::FloatExt;

mod helpers;
mod integrated;
mod spot;

#[cfg(test)]
mod tests;

/// Fourier-Malliavin volatility estimation engine.
///
/// Pre-computes the Fourier coefficients of price increments once, then
/// exposes integrated and spot estimators as cheap method calls.
///
/// Generic over `T: FloatExt` (`f32` / `f64`).
///
/// # Example
/// ```ignore
/// let engine = FMVol::new(&log_prices, &times, 1.0);
/// let iv   = engine.integrated_variance();
/// let spot = engine.spot_variance(&tau, None);
/// ```
pub struct FMVol<T: FloatExt> {
  /// Precomputed Fourier coefficients of price increments.
  pub(super) dx: Array1<Complex<T>>,
  /// Time period *T*.
  pub(super) period: T,
  /// Number of price increments (*n*).
  pub(super) n: usize,
  /// Primary cutting frequency *N*.
  pub(super) n_freq: usize,
  /// Maximum frequency stored in `dx`.
  pub(super) max_freq: usize,
}

impl<T: FloatExt> FMVol<T> {
  /// Build an engine from irregularly spaced observations.
  ///
  /// Sets `N = floor(n/2)` and pre-computes Fourier coefficients up to
  /// `N + M_max + L_max` where `M_max = floor(N^0.5)` and `L_max = floor(N^0.25)`.
  ///
  /// Panics if `prices.len() < 2` or `times.len() != prices.len()`.
  pub fn new(prices: &[T], times: &[T], period: T) -> Self {
    assert!(
      prices.len() >= 2,
      "FMVol::new requires at least 2 price observations to form increments, got {}",
      prices.len()
    );
    assert_eq!(
      prices.len(),
      times.len(),
      "FMVol::new: prices.len()={} must equal times.len()={}",
      prices.len(),
      times.len()
    );
    let n = prices.len() - 1;
    let big_n = n / 2;
    let m_max = (big_n as f64).sqrt() as usize;
    let l_max = (big_n as f64).powf(0.25) as usize;
    let max_freq = big_n + m_max + l_max;
    let dx = fourier_coefficients_dx(prices, times, period, max_freq);
    Self {
      dx,
      period,
      n,
      n_freq: big_n,
      max_freq,
    }
  }

  /// Build an engine from **uniformly spaced** observations (FFT-accelerated, O(n log n)).
  ///
  /// Assumes `t_l = l · T / n`; no explicit times array needed.
  ///
  /// Panics if `prices.len() < 2`.
  pub fn new_uniform(prices: &[T], period: T) -> Self {
    assert!(
      prices.len() >= 2,
      "FMVol::new_uniform requires at least 2 price observations to form increments, got {}",
      prices.len()
    );
    let n = prices.len() - 1;
    let big_n = n / 2;
    let m_max = (big_n as f64).sqrt() as usize;
    let l_max = (big_n as f64).powf(0.25) as usize;
    let max_freq = big_n + m_max + l_max;
    let dx = fourier_coefficients_dx_uniform(prices, period, max_freq);
    Self {
      dx,
      period,
      n,
      n_freq: big_n,
      max_freq,
    }
  }

  /// Build an engine with explicit cutting frequency *N* and maximum frequency.
  ///
  /// `max_freq` controls how high the Fourier coefficients are computed.
  /// Must satisfy `max_freq ≥ n_freq`.
  /// For spot leverage / volvol / quarticity you need `max_freq ≥ N + M + L`.
  ///
  /// Panics if `prices.len() < 2`, `times.len() != prices.len()`, or `max_freq < n_freq`.
  pub fn with_freq(prices: &[T], times: &[T], period: T, n_freq: usize, max_freq: usize) -> Self {
    assert!(
      prices.len() >= 2,
      "FMVol::with_freq requires at least 2 price observations to form increments, got {}",
      prices.len()
    );
    assert_eq!(
      prices.len(),
      times.len(),
      "FMVol::with_freq: prices.len()={} must equal times.len()={}",
      prices.len(),
      times.len()
    );
    let n = prices.len() - 1;
    assert!(
      max_freq >= n_freq,
      "max_freq={max_freq} must be ≥ n_freq={n_freq}"
    );
    let dx = fourier_coefficients_dx(prices, times, period, max_freq);
    Self {
      dx,
      period,
      n,
      n_freq,
      max_freq,
    }
  }

  /// Primary cutting frequency *N*.
  pub fn n_freq(&self) -> usize {
    self.n_freq
  }

  /// Number of price increments.
  pub fn n(&self) -> usize {
    self.n
  }

  /// Time period.
  pub fn period(&self) -> T {
    self.period
  }
}
