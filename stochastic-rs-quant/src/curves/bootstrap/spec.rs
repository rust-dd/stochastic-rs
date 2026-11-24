//! What [`bootstrap`](super::bootstrap) reads: the quotes, the interpolation and a discount curve.

use crate::curves::discount_curve::DiscountCurve;
use crate::curves::types::BootstrapInstrument;
use crate::curves::types::InterpolationMethod;
use crate::traits::RealExt;

/// The quotes a curve is bootstrapped from and how.
#[derive(Debug, Clone, Copy)]
pub struct BootstrapSpec<'a, T: RealExt> {
  pub instruments: &'a [BootstrapInstrument<T>],
  pub method: InterpolationMethod,
  /// An exogenous discount curve: the result is the tenor's pseudo-discount (forecast) curve
  /// against it.
  pub discount: Option<&'a DiscountCurve<T>>,
}

impl<'a, T: RealExt> BootstrapSpec<'a, T> {
  /// A single-curve bootstrap.
  pub fn new(instruments: &'a [BootstrapInstrument<T>], method: InterpolationMethod) -> Self {
    Self {
      instruments,
      method,
      discount: None,
    }
  }

  pub fn discounting_with(mut self, curve: &'a DiscountCurve<T>) -> Self {
    self.discount = Some(curve);
    self
  }
}
