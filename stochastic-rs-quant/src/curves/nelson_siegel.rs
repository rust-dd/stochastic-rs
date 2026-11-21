//! Nelson-Siegel parametric yield curve model.
//!
//! Reference: Nelson & Siegel, "Parsimonious Modeling of Yield Curves",
//! Journal of Business, 60(4), 473-489 (1987).
//!
//! Reference: Diebold & Li, "Forecasting the Term Structure of Government Bond Yields",
//! Journal of Econometrics, 130(2), 337-364 (2006).
//!
//! $$
//! y(\tau) = \beta_0 +
//!   \beta_1 \frac{1 - e^{-\tau/\lambda}}{\tau/\lambda} +
//!   \beta_2 \left(\frac{1 - e^{-\tau/\lambda}}{\tau/\lambda} - e^{-\tau/\lambda}\right)
//! $$

use ndarray::ArrayView1;

use super::types::CurvePoint;
use crate::QuantError;
use crate::traits::RealExt;

/// Nelson-Siegel parametric yield curve model (4 parameters).
#[derive(Debug, Clone)]
pub struct NelsonSiegel<T: RealExt> {
  /// Long-term level.
  pub beta0: T,
  /// Short-term slope.
  pub beta1: T,
  /// Medium-term curvature.
  pub beta2: T,
  /// Decay parameter.
  pub lambda: T,
}

impl<T: RealExt> NelsonSiegel<T> {
  /// Panics unless every beta is finite and `lambda` is finite and positive.
  pub fn new(beta0: T, beta1: T, beta2: T, lambda: T) -> Self {
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
      lambda.is_finite() && lambda > T::zero(),
      "lambda must satisfy `lambda.is_finite() && lambda > T::zero()`, got lambda = {lambda:?}"
    );
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

  /// Zero rate `y(τ)` of the formula in the module doc; NaN at a negative or non-finite `tau`.
  pub fn zero_rate(&self, tau: T) -> T {
    if !on_curve(tau) {
      return T::nan();
    }
    self.beta0 + self.beta1 * self.slope_loading(tau) + self.beta2 * self.curvature_loading(tau)
  }

  /// Instantaneous forward rate at `tau`; NaN at a negative or non-finite `tau`.
  ///
  /// $$
  /// f(\tau) = \beta_0 + \beta_1 e^{-\tau/\lambda} + \beta_2 \frac{\tau}{\lambda} e^{-\tau/\lambda}
  /// $$
  pub fn forward_rate(&self, tau: T) -> T {
    if !on_curve(tau) {
      return T::nan();
    }
    let x = tau / self.lambda;
    let exp_x = (-x).exp();
    self.beta0 + self.beta1 * exp_x + self.beta2 * x * exp_x
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

  /// Least squares on the betas at each decay of the grid 0.1, 0.11, …, 5.0, so a decay off it is
  /// not recovered; `Err` below four quotes or for one not finite or at a negative maturity.
  pub fn fit(maturities: ArrayView1<T>, market_rates: ArrayView1<T>) -> Result<Self, QuantError> {
    check_quotes(
      "quotes for a Nelson-Siegel fit",
      4,
      maturities,
      market_rates,
    )?;
    let n = maturities.len();
    let mut best = None::<(T, Self)>;

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

      if let Some(betas) = crate::linalg::solve_small::<3>(&xtx_data, &xty_data) {
        let candidate = Self {
          beta0: T::from_f64_fast(betas[0]),
          beta1: T::from_f64_fast(betas[1]),
          beta2: T::from_f64_fast(betas[2]),
          lambda,
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

      lambda += lambda_step;
    }
    best
      .map(|(_, curve)| curve)
      .ok_or_else(|| QuantError::Numerical {
        what: "Nelson-Siegel fit",
        reason: "no decay on the grid gives a solvable least-squares system".into(),
      })
  }
}

/// A maturity a curve has a rate at: finite and not negative, `tau = 0` at the formula's limit.
pub(super) fn on_curve<T: RealExt>(tau: T) -> bool {
  tau.is_finite() && tau >= T::zero()
}

/// `Err` unless every maturity has one rate and there are `needed` quotes, each maturity finite
/// and not negative and each rate finite.
pub(super) fn check_quotes<T: RealExt>(
  what: &'static str,
  needed: usize,
  maturities: ArrayView1<T>,
  rates: ArrayView1<T>,
) -> Result<(), QuantError> {
  if rates.len() != maturities.len() {
    return Err(QuantError::DimensionMismatch {
      what: "market rates, one per maturity",
      expected: maturities.len(),
      got: rates.len(),
    });
  }
  if maturities.len() < needed {
    return Err(QuantError::InsufficientData {
      what,
      needed,
      got: maturities.len(),
    });
  }
  for (index, (&maturity, &rate)) in maturities.iter().zip(&rates).enumerate() {
    if !on_curve(maturity) {
      return Err(QuantError::InvalidQuote {
        index,
        reason: format!(
          "maturity must satisfy `maturity.is_finite() && maturity >= T::zero()`, got maturity = {maturity:?}"
        ),
      });
    }
    if !rate.is_finite() {
      return Err(QuantError::InvalidQuote {
        index,
        reason: format!("rate must satisfy `rate.is_finite()`, got rate = {rate:?}"),
      });
    }
  }
  Ok(())
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

  #[test]
  #[should_panic(expected = "beta2 must satisfy `beta2.is_finite()`, got beta2 = inf")]
  fn new_refuses_a_beta_that_is_not_finite() {
    NelsonSiegel::new(0.04, -0.02, f64::INFINITY, 0.5);
  }

  #[test]
  #[should_panic(
    expected = "lambda must satisfy `lambda.is_finite() && lambda > T::zero()`, got lambda = 0.0"
  )]
  fn new_refuses_a_decay_that_is_not_positive() {
    NelsonSiegel::new(0.04, -0.02, 0.01, 0.0);
  }

  #[test]
  fn a_maturity_off_the_curve_is_nan_and_the_origin_its_limit() {
    let ns = NelsonSiegel::<f64>::new(0.04, -0.02, 0.01, 0.5);
    for tau in [-1.0, -1e-300, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
      let values = [
        ns.zero_rate(tau),
        ns.forward_rate(tau),
        ns.discount_factor(tau),
      ];
      assert!(values.iter().all(|v| v.is_nan()), "tau = {tau}: {values:?}");
    }
    let origin = [
      ns.zero_rate(0.0),
      ns.forward_rate(0.0),
      ns.discount_factor(0.0),
    ];
    assert_eq!(origin, [0.02, 0.02, 1.0]);
  }
}
