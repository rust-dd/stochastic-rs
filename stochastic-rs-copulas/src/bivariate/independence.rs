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
use crate::traits::bivariate::admitted;

/// No parameter, so always fitted: its `theta` is the 0 of the families it is the limit of.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Independence;

impl Independence {
  pub fn new() -> Self {
    Self
  }
}

impl BivariateExt for Independence {
  fn family(&self) -> BivariateFamily {
    BivariateFamily::Independence
  }

  fn theta(&self) -> Option<f64> {
    Some(0.0)
  }

  /// Panics unless `theta == 0`, the one value the family has.
  fn set_theta(&mut self, theta: f64) {
    self.validate_theta(theta).unwrap_or_else(|e| panic!("{e}"));
  }

  fn theta_bounds(&self) -> (f64, f64) {
    (0.0, 0.0)
  }

  fn invalid_thetas(&self) -> Vec<f64> {
    vec![]
  }

  fn validate_theta(&self, theta: f64) -> Result<(), CopulaError> {
    if theta == 0.0 {
      return Ok(());
    }
    Err(CopulaError::InvalidParameter {
      name: "theta",
      value: theta,
      constraint: "theta == 0.0".into(),
    })
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
  fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError> {
    admitted(self, tau, 0.0)
  }

  /// $C(u,v) = uv$ has no tail dependence in either tail by construction.
  fn tail_dependence(&self) -> Result<TailDependence<f64>, CopulaError> {
    Ok(TailDependence {
      lower: 0.0,
      upper: 0.0,
    })
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// No parameter to set, so a new copula is fitted already: `theta` is 0, the tails are 0 and it samples.
  #[test]
  fn independence_is_fitted_from_the_start() {
    let c = Independence::new();
    assert_eq!(c.theta(), Some(0.0));
    assert_eq!(
      c.tail_dependence(),
      Ok(TailDependence {
        lower: 0.0,
        upper: 0.0,
      })
    );
    assert_eq!(c.sample(4, &Deterministic::new(7)).unwrap().dim(), (4, 2));
  }

  #[test]
  #[should_panic(expected = "theta must satisfy `theta == 0.0`, got theta = 0.5")]
  fn independence_set_theta_admits_only_zero() {
    let mut c = Independence::new();
    c.set_theta(0.0);
    c.set_theta(0.5);
  }
}
