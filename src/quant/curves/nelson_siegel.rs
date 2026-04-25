#![allow(clippy::doc_lazy_continuation)]
//! Nelson-Siegel parametric yield curve model.
//!
//! Reference: Nelson & Siegel, "Parsimonious Modeling of Yield Curves",
//! Journal of Business, 60(4), 473-489 (1987).
//!
//! Reference: Diebold & Li, "Forecasting the Term Structure of Government Bond Yields",
//! Journal of Econometrics, 130(2), 337-364 (2006).
//!
//! $$
//! y(\tau) = \beta_0
//!   + \beta_1 \frac{1 - e^{-\tau/\lambda}}{\tau/\lambda}
//!   + \beta_2 \left(\frac{1 - e^{-\tau/\lambda}}{\tau/\lambda} - e^{-\tau/\lambda}\right)
//! $$

use ndarray::Array1;

use super::types::CurvePoint;
use crate::traits::FloatExt;

/// Nelson-Siegel parametric yield curve model (4 parameters).
#[derive(Debug, Clone)]
pub struct NelsonSiegel<T: FloatExt> {
  /// Long-term level.
  pub beta0: T,
  /// Short-term slope.
  pub beta1: T,
  /// Medium-term curvature.
  pub beta2: T,
  /// Decay parameter.
  pub lambda: T,
}

impl<T: FloatExt> NelsonSiegel<T> {
  pub fn new(beta0: T, beta1: T, beta2: T, lambda: T) -> Self {
    Self {
      beta0,
      beta1,
      beta2,
      lambda,
    }
  }

  /// Factor loading for the slope component.
  pub(crate) fn slope_loading(&self, tau: T) -> T {
    let x = tau / self.lambda;
    if x < T::from_f64_fast(1e-10) {
      return T::one();
    }
    (T::one() - (-x).exp()) / x
  }

  /// Factor loading for the curvature component.
  pub(crate) fn curvature_loading(&self, tau: T) -> T {
    let x = tau / self.lambda;
    if x < T::from_f64_fast(1e-10) {
      return T::zero();
    }
    (T::one() - (-x).exp()) / x - (-x).exp()
  }

  /// Compute the zero rate at maturity `tau`.
  pub fn zero_rate(&self, tau: T) -> T {
    self.beta0 + self.beta1 * self.slope_loading(tau) + self.beta2 * self.curvature_loading(tau)
  }

  /// Compute the instantaneous forward rate at maturity `tau`.
  ///
  /// $$
  /// f(\tau) = \beta_0 + \beta_1 e^{-\tau/\lambda} + \beta_2 \frac{\tau}{\lambda} e^{-\tau/\lambda}
  /// $$
  pub fn forward_rate(&self, tau: T) -> T {
    let x = tau / self.lambda;
    let exp_x = (-x).exp();
    self.beta0 + self.beta1 * exp_x + self.beta2 * x * exp_x
  }

  /// Compute the discount factor at maturity `tau` (continuous compounding).
  pub fn discount_factor(&self, tau: T) -> T {
    (-self.zero_rate(tau) * tau).exp()
  }

  /// Generate curve points at the given maturities.
  pub fn curve_points(&self, maturities: &Array1<T>) -> Vec<CurvePoint<T>> {
    maturities
      .iter()
      .map(|&tau| CurvePoint {
        time: tau,
        discount_factor: self.discount_factor(tau),
      })
      .collect()
  }

  /// Fit the model to market zero rates using grid search on lambda + OLS for betas (Diebold-Li).
  #[cfg(feature = "openblas")]
  pub fn fit(maturities: &Array1<T>, market_rates: &Array1<T>) -> Self {
    let n = maturities.len();
    let mut best_sse = T::from_f64_fast(f64::MAX);
    let mut best = Self::new(T::zero(), T::zero(), T::zero(), T::one());

    let lambda_min = T::from_f64_fast(0.1);
    let lambda_max = T::from_f64_fast(5.0);
    let lambda_step = T::from_f64_fast(0.01);

    let mut lambda = lambda_min;
    while lambda <= lambda_max {
      let trial = Self::new(T::zero(), T::zero(), T::zero(), lambda);

      let mut xtx_data = [0.0_f64; 9];
      let mut xty_data = [0.0_f64; 3];

      for i in 0..n {
        let tau = maturities[i];
        let y = market_rates[i].to_f64().unwrap();
        let x = [
          1.0,
          trial.slope_loading(tau).to_f64().unwrap(),
          trial.curvature_loading(tau).to_f64().unwrap(),
        ];

        for r in 0..3 {
          xty_data[r] += x[r] * y;
          for c in 0..3 {
            xtx_data[r * 3 + c] += x[r] * x[c];
          }
        }
      }

      if let Some(betas) = super::linalg::solve_linalg::<3>(&xtx_data, &xty_data) {
        let candidate = Self::new(
          T::from_f64_fast(betas[0]),
          T::from_f64_fast(betas[1]),
          T::from_f64_fast(betas[2]),
          lambda,
        );
        let mut sse = T::zero();
        for i in 0..n {
          let err = market_rates[i] - candidate.zero_rate(maturities[i]);
          sse += err * err;
        }
        if sse < best_sse {
          best_sse = sse;
          best = candidate;
        }
      }

      lambda += lambda_step;
    }
    best
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn long_rate_approaches_beta0() {
    // y(τ→∞) → β₀
    let ns = NelsonSiegel::<f64>::new(0.04, -0.02, 0.01, 0.5);
    let r_long = ns.zero_rate(50.0);
    assert!((r_long - 0.04).abs() < 1e-3, "{r_long} ≠ β0=0.04");
  }

  #[test]
  fn short_rate_equals_beta0_plus_beta1() {
    // y(τ→0) → β₀ + β₁
    let ns = NelsonSiegel::<f64>::new(0.04, -0.02, 0.01, 0.5);
    let r_short = ns.zero_rate(1e-6);
    assert!((r_short - 0.02).abs() < 1e-3, "{r_short} ≠ β0+β1=0.02");
  }

  #[test]
  fn discount_factor_at_zero_is_one() {
    let ns = NelsonSiegel::<f64>::new(0.04, -0.02, 0.01, 0.5);
    let df = ns.discount_factor(0.0);
    assert!((df - 1.0).abs() < 1e-12);
  }
}
