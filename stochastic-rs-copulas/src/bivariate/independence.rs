//! # Independence
//!
//! $$
//! C(u,v)=uv
//! $$
//!

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::Axis;

use super::BivariateFamily;
use crate::bivariate::conditional::conditional_cdf;
use crate::bivariate::conditional::conditional_quantiles;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;

#[derive(Debug, Clone)]
pub struct Independence {
  pub r#type: BivariateFamily,
  pub theta: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Default for Independence {
  fn default() -> Self {
    Self {
      r#type: BivariateFamily::Independence,
      theta: None,
      theta_bounds: (0.0, 0.0),
      invalid_thetas: vec![],
    }
  }
}

impl Independence {
  pub fn new() -> Self {
    Self::default()
  }
}

impl BivariateExt for Independence {
  fn family(&self) -> BivariateFamily {
    self.r#type
  }

  fn theta(&self) -> Option<f64> {
    self.theta
  }

  fn theta_bounds(&self) -> (f64, f64) {
    self.theta_bounds
  }

  fn invalid_thetas(&self) -> Vec<f64> {
    self.invalid_thetas.clone()
  }

  fn set_theta(&mut self, theta: f64) {
    self.theta = Some(theta);
  }

  fn generator(&self, t: ArrayView1<f64>) -> Result<Array1<f64>, CopulaError> {
    Ok(t.ln())
  }

  fn pdf(&self, X: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    let in_range = X.map_axis(Axis(1), |row| {
      row.iter().all(|&val| (0.0..=1.0).contains(&val))
    });

    let out = in_range.map(|&val| if val { 1.0 } else { 0.0 });

    Ok(out)
  }

  fn cdf(&self, X: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    let U = X.column(0);
    let V = X.column(1);

    Ok(&U * &V)
  }

  /// `∂_v (uv) = u`, NaN for `v` outside `[0, 1]`.
  fn partial_derivative(&self, X: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    Ok(conditional_cdf(X, |u, _| u))
  }

  /// The identity, NaN for `y` or `v` outside `[0, 1]`.
  fn percent_point(
    &self,
    y: ArrayView1<f64>,
    V: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    conditional_quantiles(y, V, |y, _| y)
  }

  fn tau_from_theta(&self, _theta: f64) -> f64 {
    0.0
  }

  /// The family represents τ = 0 only; a fit does not reject other data.
  fn theta_from_tau(&self, _tau: f64) -> Result<f64, CopulaError> {
    Ok(0.0)
  }

  /// $C(u,v) = uv$ has no tail dependence in either tail by construction.
  fn tail_dependence(&self) -> TailDependence<f64> {
    self.assert_theta_valid_for_tail_dependence();
    TailDependence {
      lower: 0.0,
      upper: 0.0,
    }
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  #[test]
  fn independence_tail_dependence_is_zero() {
    let mut c = Independence::new();
    c.set_theta(0.0);
    let td = c.tail_dependence();
    assert_eq!(td.lower, 0.0);
    assert_eq!(td.upper, 0.0);
  }

  #[test]
  #[should_panic(expected = "tail_dependence requires a valid theta")]
  fn independence_tail_dependence_panics_when_theta_unset() {
    let _ = Independence::new().tail_dependence();
  }

  #[test]
  fn independence_samples_through_the_conditional_sampler_once_theta_is_zero() {
    let mut c = Independence::new();
    assert_eq!(
      c.sample(4, &Deterministic::new(7)).unwrap_err(),
      CopulaError::NotFitted
    );
    c.set_theta(0.0);
    assert_eq!(c.sample(4, &Deterministic::new(7)).unwrap().dim(), (4, 2));
  }
}
