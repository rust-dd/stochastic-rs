#![allow(clippy::doc_lazy_continuation)]
//! Nelson-Siegel-Svensson parametric yield curve model.
//!
//! Reference: Svensson, "Estimating and Interpreting Forward Interest Rates: Sweden 1992-1994",
//! IMF Working Paper 94/114 (1994). Also: BIS Papers No 25.
//!
//! $$
//! y(\tau) = \beta_0
//!   + \beta_1 \frac{1 - e^{-\tau/\lambda_1}}{\tau/\lambda_1}
//!   + \beta_2 \left(\frac{1 - e^{-\tau/\lambda_1}}{\tau/\lambda_1} - e^{-\tau/\lambda_1}\right)
//!   + \beta_3 \left(\frac{1 - e^{-\tau/\lambda_2}}{\tau/\lambda_2} - e^{-\tau/\lambda_2}\right)
//! $$

use ndarray::Array1;

use super::types::CurvePoint;
use crate::traits::FloatExt;

/// Nelson-Siegel-Svensson parametric yield curve model (6 parameters).
#[derive(Debug, Clone)]
pub struct Svensson<T: FloatExt> {
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

impl<T: FloatExt> Svensson<T> {
  pub fn new(beta0: T, beta1: T, beta2: T, beta3: T, lambda1: T, lambda2: T) -> Self {
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

  /// Compute the zero rate at maturity `tau`.
  pub fn zero_rate(&self, tau: T) -> T {
    self.beta0
      + self.beta1 * self.slope_loading(tau, self.lambda1)
      + self.beta2 * self.curvature_loading(tau, self.lambda1)
      + self.beta3 * self.curvature_loading(tau, self.lambda2)
  }

  /// Compute the instantaneous forward rate at maturity `tau`.
  ///
  /// $f(\tau) = \beta_0 + \beta_1 e^{-\tau/\lambda_1} + \beta_2 \frac{\tau}{\lambda_1} e^{-\tau/\lambda_1} + \beta_3 \frac{\tau}{\lambda_2} e^{-\tau/\lambda_2}$
  pub fn forward_rate(&self, tau: T) -> T {
    let x1 = tau / self.lambda1;
    let x2 = tau / self.lambda2;
    let exp_x1 = (-x1).exp();
    let exp_x2 = (-x2).exp();
    self.beta0 + self.beta1 * exp_x1 + self.beta2 * x1 * exp_x1 + self.beta3 * x2 * exp_x2
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

  /// Fit the model to market zero rates using grid search on (lambda1, lambda2) + OLS for betas.
  pub fn fit(maturities: &Array1<T>, market_rates: &Array1<T>) -> Self {
    let n = maturities.len();
    let mut best_sse = T::from_f64_fast(f64::MAX);
    let mut best = Self::new(
      T::zero(),
      T::zero(),
      T::zero(),
      T::zero(),
      T::one(),
      T::from_f64_fast(2.0),
    );

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

        if let Some(betas) = super::linalg::solve_linalg::<4>(&xtx_data, &xty_data) {
          let candidate = Self::new(
            T::from_f64_fast(betas[0]),
            T::from_f64_fast(betas[1]),
            T::from_f64_fast(betas[2]),
            T::from_f64_fast(betas[3]),
            l1,
            l2,
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

        l2 += step;
      }
      l1 += step;
    }
    best
  }
}
