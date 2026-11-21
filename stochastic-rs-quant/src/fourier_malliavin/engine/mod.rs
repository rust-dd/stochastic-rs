//! The Fourier–Malliavin engine: raw estimators in the FMVol MATLAB conventions, and the
//! bias-corrected vol-of-vol estimators of Toscano et al., eq. (4), (11) and (51), for any period.
//!
//! Reference: Sanfelici & Toscano (2024), arXiv:2402.00172
//! Reference: Malliavin & Mancino (2002), Finance and Stochastics 6(1); (2009), Ann. Statist. 37(4)
//! Reference: Mancino & Recchioni (2015)
//! Reference: Toscano, Livieri, Mancino & Marmi (2022), arXiv:2112.14529v3

use ndarray::Array1;
use num_complex::Complex;

use super::coefficients::fourier_coefficients_dx;
use super::coefficients::fourier_coefficients_dx_uniform;
use crate::QuantError;
use crate::traits::RealExt;

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

/// Fourier–Malliavin engine: the Fourier coefficients of the price increments are computed once
/// and every integrated and spot estimator reads them.
///
/// ```
/// use stochastic_rs_quant::fourier_malliavin::FmVol;
///
/// let n = 200;
/// let times: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
/// let log_prices: Vec<f64> = times.iter().map(|&t| (10.0 * t).sin() * 0.01).collect();
/// let engine = FmVol::new(&log_prices, &times, 1.0);
///
/// let iv = engine.integrated_variance();
/// let tau = [0.25, 0.5, 0.75];
/// let spot = engine.spot_variance(&tau, None);
/// assert!(iv.is_finite());
/// assert_eq!(spot.len(), 3);
/// ```
pub struct FmVol<T: RealExt> {
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

impl<T: RealExt> FmVol<T> {
  /// An engine on irregularly spaced observations with `N = floor(n/2)`; panics where
  /// [`Self::try_new`] returns an error.
  pub fn new(prices: &[T], times: &[T], period: T) -> Self {
    Self::try_new(prices, times, period).unwrap_or_else(|e| panic!("{e}"))
  }

  /// `Err` unless prices and times are finite, times strictly increase and span `period`, and
  /// the default frequency windows lie below the discrete Nyquist storage bound.
  pub fn try_new(prices: &[T], times: &[T], period: T) -> Result<Self, QuantError> {
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

  /// An engine on uniformly spaced observations `t_l = l · T / n`, by FFT in O(n log n); panics
  /// where [`Self::try_new_uniform`] returns an error.
  pub fn new_uniform(prices: &[T], period: T) -> Self {
    Self::try_new_uniform(prices, period).unwrap_or_else(|e| panic!("{e}"))
  }

  /// Fallible variant of [`Self::new_uniform`].
  pub fn try_new_uniform(prices: &[T], period: T) -> Result<Self, QuantError> {
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

  /// An engine with an explicit cutting frequency `N` and `max_freq ≥ n_freq` (spot leverage,
  /// vol of vol and quarticity need `N + M + L`); panics where [`Self::try_with_freq`] errs.
  pub fn with_freq(prices: &[T], times: &[T], period: T, n_freq: usize, max_freq: usize) -> Self {
    Self::try_with_freq(prices, times, period, n_freq, max_freq).unwrap_or_else(|e| panic!("{e}"))
  }

  /// Fallible variant of [`Self::with_freq`].
  pub fn try_with_freq(
    prices: &[T],
    times: &[T],
    period: T,
    n_freq: usize,
    max_freq: usize,
  ) -> Result<Self, QuantError> {
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
