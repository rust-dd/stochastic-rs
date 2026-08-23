//! Short-rate zero-coupon bond pricing — `ShortRatePricer`.

/// Zero-coupon bond pricing from a short-rate model.
///
/// The model holds its own parameters; the short rate and maturity are the
/// query. This is the split that makes one model reusable across a maturity
/// grid — `bonds::Cir` previously stored `r_t` and `tau` as fields, so a
/// second maturity meant a second struct.
pub trait ShortRatePricer {
  /// Price of a zero-coupon bond paying 1 at `tau`, given short rate `r0`.
  ///
  /// # Panics
  /// - if `tau` is negative
  fn zero_coupon_price(&self, r0: f64, tau: f64) -> f64;

  /// Continuously-compounded zero yield implied by
  /// [`zero_coupon_price`](Self::zero_coupon_price).
  ///
  /// Returns `NaN` at `tau == 0`, where the yield is undefined — the price
  /// is 1 and the log-ratio is 0/0.
  fn zero_yield(&self, r0: f64, tau: f64) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    -self.zero_coupon_price(r0, tau).ln() / tau
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::bonds::cir::Cir;

  #[test]
  fn zero_yield_is_nan_at_zero_tau() {
    let c = Cir {
      theta: 0.5,
      mu: 0.04,
      sigma: 0.01,
    };
    assert!(c.zero_yield(0.05, 0.0).is_nan());
  }

  #[test]
  fn zero_yield_is_finite_for_positive_tau() {
    let c = Cir {
      theta: 0.5,
      mu: 0.04,
      sigma: 0.01,
    };
    assert!(c.zero_yield(0.05, 1.0).is_finite());
  }
}
