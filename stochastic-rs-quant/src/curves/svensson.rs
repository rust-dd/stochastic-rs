//! Nelson-Siegel-Svensson parametric yield curve model.
//!
//! Reference: Svensson (1994), "Estimating and Interpreting Forward Interest Rates", IMF WP 94/114
//! Reference: BIS Papers No. 25 (2005), <https://www.bis.org/publ/bppdf/bispap25.htm>
//!
//! $$
//! y(\tau) = \beta_0 +
//!   \beta_1 \frac{1 - e^{-\tau/\lambda_1}}{\tau/\lambda_1} +
//!   \beta_2 \left(\frac{1 - e^{-\tau/\lambda_1}}{\tau/\lambda_1} - e^{-\tau/\lambda_1}\right) +
//!   \beta_3 \left(\frac{1 - e^{-\tau/\lambda_2}}{\tau/\lambda_2} - e^{-\tau/\lambda_2}\right)
//! $$

use ndarray::ArrayView1;

use super::nelson_siegel::check_quotes;
use super::nelson_siegel::on_curve;
use super::types::CurvePoint;
use crate::QuantError;
use crate::traits::RealExt;

/// Nelson-Siegel-Svensson parametric yield curve model (6 parameters).
#[derive(Debug, Clone)]
pub struct Svensson<T: RealExt> {
  /// Long-term level.
  pub beta0: T,
  /// Short-term slope.
  pub beta1: T,
  /// First curvature.
  pub beta2: T,
  /// Second curvature.
  pub beta3: T,
  /// First decay parameter.
  pub lambda1: T,
  /// Second decay parameter.
  pub lambda2: T,
}

impl<T: RealExt> Svensson<T> {
  /// Panics unless every beta is finite and both decays are finite and positive.
  pub fn new(beta0: T, beta1: T, beta2: T, beta3: T, lambda1: T, lambda2: T) -> Self {
    assert!(
      beta0.is_finite(),
      "beta0 must satisfy `beta0.is_finite()`, got beta0 = {beta0:?}"
    );
    assert!(
      beta1.is_finite(),
      "beta1 must satisfy `beta1.is_finite()`, got beta1 = {beta1:?}"
    );
    assert!(
      beta2.is_finite(),
      "beta2 must satisfy `beta2.is_finite()`, got beta2 = {beta2:?}"
    );
    assert!(
      beta3.is_finite(),
      "beta3 must satisfy `beta3.is_finite()`, got beta3 = {beta3:?}"
    );
    assert!(
      lambda1.is_finite() && lambda1 > T::zero(),
      "lambda1 must satisfy `lambda1.is_finite() && lambda1 > T::zero()`, got lambda1 = {lambda1:?}"
    );
    assert!(
      lambda2.is_finite() && lambda2 > T::zero(),
      "lambda2 must satisfy `lambda2.is_finite() && lambda2 > T::zero()`, got lambda2 = {lambda2:?}"
    );
    Self {
      beta0,
      beta1,
      beta2,
      beta3,
      lambda1,
      lambda2,
    }
  }

  fn slope_loading(&self, tau: T, lambda: T) -> T {
    let x = tau / lambda;
    if x < T::from_f64_fast(1e-10) {
      return T::one();
    }
    (T::one() - (-x).exp()) / x
  }

  fn curvature_loading(&self, tau: T, lambda: T) -> T {
    let x = tau / lambda;
    if x < T::from_f64_fast(1e-10) {
      return T::zero();
    }
    (T::one() - (-x).exp()) / x - (-x).exp()
  }

  /// Zero rate `y(τ)` of the formula in the module doc; NaN at a negative or non-finite `tau`.
  pub fn zero_rate(&self, tau: T) -> T {
    if !on_curve(tau) {
      return T::nan();
    }
    self.beta0
      + self.beta1 * self.slope_loading(tau, self.lambda1)
      + self.beta2 * self.curvature_loading(tau, self.lambda1)
      + self.beta3 * self.curvature_loading(tau, self.lambda2)
  }

  /// Instantaneous forward rate at `tau`; NaN at a negative or non-finite `tau`.
  ///
  /// $$
  /// f(\tau) = \beta_0 + \beta_1 e^{-\tau/\lambda_1} + \beta_2 \frac{\tau}{\lambda_1} e^{-\tau/\lambda_1} +
  ///   \beta_3 \frac{\tau}{\lambda_2} e^{-\tau/\lambda_2}
  /// $$
  pub fn forward_rate(&self, tau: T) -> T {
    if !on_curve(tau) {
      return T::nan();
    }
    let x1 = tau / self.lambda1;
    let x2 = tau / self.lambda2;
    let exp_x1 = (-x1).exp();
    let exp_x2 = (-x2).exp();
    self.beta0 + self.beta1 * exp_x1 + self.beta2 * x1 * exp_x1 + self.beta3 * x2 * exp_x2
  }

  /// `e^{−τ y(τ)}`, 1 at `tau = 0`; NaN at a negative or non-finite `tau`.
  pub fn discount_factor(&self, tau: T) -> T {
    (-self.zero_rate(tau) * tau).exp()
  }

  /// Generate curve points at the given maturities.
  pub fn curve_points(&self, maturities: ArrayView1<T>) -> Vec<CurvePoint<T>> {
    maturities
      .iter()
      .map(|&tau| CurvePoint {
        time: tau,
        discount_factor: self.discount_factor(tau),
      })
      .collect()
  }

  /// Least squares on the betas at each decay pair of the grid 0.1, 0.15, …, 5.0, so a decay off
  /// it is not recovered; `Err` below six quotes or for one not finite or at a negative maturity.
  pub fn fit(maturities: ArrayView1<T>, market_rates: ArrayView1<T>) -> Result<Self, QuantError> {
    check_quotes("quotes for a Svensson fit", 6, maturities, market_rates)?;
    let n = maturities.len();
    let mut best = None::<(T, Self)>;

    let lambda_min = T::from_f64_fast(0.1);
    let lambda_max = T::from_f64_fast(5.0);
    let step = T::from_f64_fast(0.05);

    let mut l1 = lambda_min;
    while l1 <= lambda_max {
      let mut l2 = lambda_min;
      while l2 <= lambda_max {
        if (l1 - l2).abs() < T::from_f64_fast(0.05) {
          l2 += step;
          continue;
        }

        let trial = Self::new(T::zero(), T::zero(), T::zero(), T::zero(), l1, l2);
        let mut xtx_data = [0.0_f64; 16];
        let mut xty_data = [0.0_f64; 4];

        for i in 0..n {
          let tau = maturities[i];
          let y = market_rates[i].to_f64().unwrap();
          let x = [
            1.0,
            trial.slope_loading(tau, l1).to_f64().unwrap(),
            trial.curvature_loading(tau, l1).to_f64().unwrap(),
            trial.curvature_loading(tau, l2).to_f64().unwrap(),
          ];

          for r in 0..4 {
            xty_data[r] += x[r] * y;
            for c in 0..4 {
              xtx_data[r * 4 + c] += x[r] * x[c];
            }
          }
        }

        if let Some(betas) = crate::linalg::solve_small::<4>(&xtx_data, &xty_data) {
          let candidate = Self {
            beta0: T::from_f64_fast(betas[0]),
            beta1: T::from_f64_fast(betas[1]),
            beta2: T::from_f64_fast(betas[2]),
            beta3: T::from_f64_fast(betas[3]),
            lambda1: l1,
            lambda2: l2,
          };
          let mut sse = T::zero();
          for i in 0..n {
            let err = market_rates[i] - candidate.zero_rate(maturities[i]);
            sse += err * err;
          }
          if sse.is_finite() && best.as_ref().is_none_or(|(least, _)| sse < *least) {
            best = Some((sse, candidate));
          }
        }

        l2 += step;
      }
      l1 += step;
    }
    best
      .map(|(_, curve)| curve)
      .ok_or_else(|| QuantError::Numerical {
        what: "Svensson fit",
        reason: "no decay pair on the grid gives a solvable least-squares system".into(),
      })
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn long_rate_approaches_beta0() {
    let s = Svensson::<f64>::new(0.04, -0.02, 0.01, 0.005, 0.5, 2.0);
    let r_long = s.zero_rate(50.0);
    assert!((r_long - 0.04).abs() < 1e-3, "{r_long} ≠ β0=0.04");
  }

  #[test]
  fn discount_factor_at_zero_is_one() {
    let s = Svensson::<f64>::new(0.04, -0.02, 0.01, 0.005, 0.5, 2.0);
    let df = s.discount_factor(0.0);
    assert!((df - 1.0).abs() < 1e-12);
  }

  #[test]
  #[should_panic(expected = "beta0 must satisfy `beta0.is_finite()`, got beta0 = NaN")]
  fn new_refuses_a_beta_that_is_not_finite() {
    Svensson::new(f64::NAN, -0.02, 0.01, 0.005, 0.5, 2.0);
  }

  #[test]
  #[should_panic(
    expected = "lambda2 must satisfy `lambda2.is_finite() && lambda2 > T::zero()`, got lambda2 = inf"
  )]
  fn new_refuses_a_decay_that_is_not_finite() {
    Svensson::new(0.04, -0.02, 0.01, 0.005, 0.5, f64::INFINITY);
  }

  #[test]
  fn a_maturity_off_the_curve_is_nan_and_the_origin_its_limit() {
    let s = Svensson::<f64>::new(0.04, -0.02, 0.01, 0.005, 0.5, 2.0);
    for tau in [-1.0, -1e-300, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
      let values = [
        s.zero_rate(tau),
        s.forward_rate(tau),
        s.discount_factor(tau),
      ];
      assert!(values.iter().all(|v| v.is_nan()), "tau = {tau}: {values:?}");
    }
    let origin = [
      s.zero_rate(0.0),
      s.forward_rate(0.0),
      s.discount_factor(0.0),
    ];
    assert_eq!(origin, [0.02, 0.02, 1.0]);
  }
}
