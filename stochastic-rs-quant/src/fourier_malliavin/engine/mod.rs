//! Struct-based Fourier-Malliavin volatility estimation engine.
//!
//! References:
//! - Sanfelici & Toscano (2024), arXiv:2402.00172.
//! - Malliavin & Mancino (2002, 2009).
//! - Mancino & Recchioni (2015).
//! - Toscano, Livieri, Mancino & Marmi (2022), arXiv:2112.14529v3.
//!
//! The raw estimators retain the FMVol MATLAB conventions. The bias-corrected
//! volatility-of-volatility estimators implement equations (4), (11), and
//! (51) of Toscano et al., including the coefficient-level quarticity
//! correction and the normalization for an arbitrary observation period.

use ndarray::Array1;
use num_complex::Complex;

use super::coefficients::fourier_coefficients_dx;
use super::coefficients::fourier_coefficients_dx_uniform;
use crate::traits::FloatExt;

mod helpers;
mod integrated;
mod spot;
mod validation;

#[cfg(test)]
mod bias_correction_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod validation_tests;

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
  /// Largest observation interval.
  pub(super) mesh: T,
  /// Origin of the Fourier time coordinate.
  pub(super) origin: T,
  /// Primary cutting frequency *N*.
  pub(super) n_freq: usize,
  /// Maximum frequency stored in `dx`.
  pub(super) max_freq: usize,
}

impl<T: FloatExt> FMVol<T> {
  /// Build an engine from irregularly spaced observations.
  ///
  /// Sets `N = floor(n/2)` and pre-computes every frequency needed by the
  /// default raw and bias-corrected windows.
  ///
  /// Panics when input or default-frequency validation fails. Use
  /// [`Self::try_new`] to receive the validation error.
  pub fn new(prices: &[T], times: &[T], period: T) -> Self {
    Self::try_new(prices, times, period)
      .expect("FMVol::new precondition violated — call try_new to handle this gracefully")
  }

  /// Fallible variant of [`Self::new`].
  ///
  /// Prices and times must be finite, times must be strictly increasing and
  /// span `period`, and the default frequency windows must lie below the
  /// discrete Nyquist storage bound.
  pub fn try_new(prices: &[T], times: &[T], period: T) -> anyhow::Result<Self> {
    let (n, mesh, origin) = validation::validate_irregular_inputs(prices, times, period)?;
    let big_n = n / 2;
    let max_freq = validation::default_max_frequency(n, big_n, mesh)?;
    validation::validate_frequency_bounds(n, big_n, max_freq)?;
    let dx = fourier_coefficients_dx(prices, times, period, max_freq);
    Ok(Self {
      dx,
      period,
      n,
      mesh,
      origin,
      n_freq: big_n,
      max_freq,
    })
  }

  /// Build an engine from **uniformly spaced** observations (FFT-accelerated, O(n log n)).
  ///
  /// Assumes `t_l = l · T / n`; no explicit times array needed.
  ///
  /// Panics when input or default-frequency validation fails. Use
  /// [`Self::try_new_uniform`] to receive the validation error.
  pub fn new_uniform(prices: &[T], period: T) -> Self {
    Self::try_new_uniform(prices, period).expect(
      "FMVol::new_uniform precondition violated — call try_new_uniform to handle this gracefully",
    )
  }

  /// Fallible variant of [`Self::new_uniform`].
  pub fn try_new_uniform(prices: &[T], period: T) -> anyhow::Result<Self> {
    let n = validation::validate_uniform_inputs(prices, period)?;
    let mesh = period / T::from_usize_(n);
    let big_n = n / 2;
    let max_freq = validation::default_max_frequency(n, big_n, mesh)?;
    validation::validate_frequency_bounds(n, big_n, max_freq)?;
    let dx = fourier_coefficients_dx_uniform(prices, period, max_freq);
    Ok(Self {
      dx,
      period,
      n,
      mesh,
      origin: T::zero(),
      n_freq: big_n,
      max_freq,
    })
  }

  /// Build an engine with explicit cutting frequency *N* and maximum frequency.
  ///
  /// `max_freq` controls how high the Fourier coefficients are computed.
  /// Must satisfy `max_freq ≥ n_freq`.
  /// For spot leverage / volvol / quarticity you need `max_freq ≥ N + M + L`.
  ///
  /// Panics when input or frequency validation fails. Use
  /// [`Self::try_with_freq`] to receive the validation error.
  pub fn with_freq(prices: &[T], times: &[T], period: T, n_freq: usize, max_freq: usize) -> Self {
    Self::try_with_freq(prices, times, period, n_freq, max_freq).expect(
      "FMVol::with_freq precondition violated — call try_with_freq to handle this gracefully",
    )
  }

  /// Fallible variant of [`Self::with_freq`].
  pub fn try_with_freq(
    prices: &[T],
    times: &[T],
    period: T,
    n_freq: usize,
    max_freq: usize,
  ) -> anyhow::Result<Self> {
    let (n, mesh, origin) = validation::validate_irregular_inputs(prices, times, period)?;
    validation::validate_frequency_bounds(n, n_freq, max_freq)?;
    let dx = fourier_coefficients_dx(prices, times, period, max_freq);
    Ok(Self {
      dx,
      period,
      n,
      mesh,
      origin,
      n_freq,
      max_freq,
    })
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

  /// Largest gap between consecutive observations.
  pub fn mesh(&self) -> T {
    self.mesh
  }

  /// Origin used by the Fourier phase convention.
  pub fn time_origin(&self) -> T {
    self.origin
  }
}
