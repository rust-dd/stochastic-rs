//! What [`bootstrap`](super::bootstrap) reads: the quotes, the interpolation and its two options.

use crate::curves::discount_curve::DiscountCurve;
use crate::curves::types::BootstrapInstrument;
use crate::curves::types::InterpolationMethod;
use crate::traits::RealExt;

/// Stopping rule of the bisection on a swap's last discount factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IterativeOptions<T: RealExt> {
  /// Largest accepted error in the repriced swap rate.
  pub tolerance: T,
  pub max_iterations: usize,
}

impl<T: RealExt> Default for IterativeOptions<T> {
  fn default() -> Self {
    Self {
      tolerance: T::from_f64_fast(1e-12),
      max_iterations: 100,
    }
  }
}

/// The quotes a curve is bootstrapped from and how.
#[derive(Debug, Clone, Copy)]
pub struct BootstrapSpec<'a, T: RealExt> {
  pub instruments: &'a [BootstrapInstrument<T>],
  pub method: InterpolationMethod,
  /// Bisection in `(0, 1)` when a `Swap`'s direct solve leaves that interval; `None` keeps the
  /// direct solve.
  pub iterative: Option<IterativeOptions<T>>,
  /// An exogenous discount curve: the result is the tenor's pseudo-discount (forecast) curve
  /// against it.
  pub discount: Option<&'a DiscountCurve<T>>,
}

impl<'a, T: RealExt> BootstrapSpec<'a, T> {
  /// A single-curve bootstrap with the direct solve.
  pub fn new(instruments: &'a [BootstrapInstrument<T>], method: InterpolationMethod) -> Self {
    Self {
      instruments,
      method,
      iterative: None,
      discount: None,
    }
  }

  pub fn iterative(mut self, options: IterativeOptions<T>) -> Self {
    self.iterative = Some(options);
    self
  }

  pub fn discounting_with(mut self, curve: &'a DiscountCurve<T>) -> Self {
    self.discount = Some(curve);
    self
  }
}
