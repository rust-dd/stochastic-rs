//! # Farlie-Gumbel-Morgenstern (FGM) copula
//!
//! $$
//! C_\theta(u,v) = uv + \theta\, u(1-u)\, v(1-v), \qquad \theta \in [-1, 1].
//! $$
//!
//! Symmetric perturbation of the independence copula with no tail dependence
//! (zero in either tail). Dependence is mild — Kendall's tau is bounded by
//! $|\tau| \le 2/9 \approx 0.222$ — so FGM is appropriate for weakly
//! dependent margins (residual cross-section correlation, etc.) and as a
//! building block for finite-mixture copulas.
//!
//! Reference: Farlie, D.J.G. (1960), "The Performance of Some Correlation
//! Coefficients for a General Bivariate Distribution", *Biometrika* 47, 307-323.
//! Reference: Nelsen, R.B. (2006), "An Introduction to Copulas", 2nd ed.,
//! Springer, Example 3.12.
//! Reference: Johnson, M.E. (1987), "Multivariate Statistical Simulation", Wiley, the Morgenstern generator (the conditional inverse), DOI 10.1002/9781118150740.

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;

use crate::bivariate::BivariateFamily;
use crate::bivariate::conditional::conditional_cdf;
use crate::bivariate::conditional::conditional_quantiles;
use crate::bivariate::conditional::copula_cdf;
use crate::bivariate::conditional::copula_density;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::admitted;
use crate::traits::bivariate::tau_out_of_range;

#[derive(Debug, Clone, Default)]
pub struct Fgm {
  theta: Option<f64>,
}

impl Fgm {
  /// An FGM copula; panics unless `-1 <= theta <= 1`.
  pub fn new(theta: f64) -> Self {
    Self::try_new(theta).unwrap_or_else(|e| panic!("{e}"))
  }

  /// `new` with the domain error returned instead of a panic.
  pub fn try_new(theta: f64) -> Result<Self, CopulaError> {
    let copula = Self { theta: Some(theta) };
    copula.check_theta()?;
    Ok(copula)
  }

  /// The copula whose Kendall's τ is `tau`.
  pub fn from_tau(tau: f64) -> Result<Self, CopulaError> {
    Self::try_new(Self::default().theta_from_tau(tau)?)
  }

  /// The root in `[0, 1]` of `u(1 + a) − au² = y`, `a = θ(1 − 2v)`, as `2y/((1 + a) + √D)` with `D` summed from
  /// non-negative terms for either sign of `a`; 0 at level 0, where `a = −1` would divide 0 by 0.
  pub(crate) fn inverse(theta: f64) -> impl Fn(f64, f64) -> f64 {
    move |y, v| {
      let a = theta * (1.0 - 2.0 * v);
      let discriminant = if a > 0.0 {
        (1.0 - a) * (1.0 - a) + 4.0 * a * (1.0 - y)
      } else {
        (1.0 + a) * (1.0 + a) - 4.0 * a * y
      };
      // `(1 + a) + √D` vanishes only at level 0 on `a = −1`; the floor answers 0 there without a branch.
      2.0 * y / ((1.0 + a) + discriminant.sqrt()).max(f64::MIN_POSITIVE)
    }
  }
}

impl BivariateExt for Fgm {
  fn family(&self) -> BivariateFamily {
    BivariateFamily::Fgm
  }

  fn theta(&self) -> Option<f64> {
    self.theta
  }

  fn set_theta(&mut self, theta: f64) {
    self.validate_theta(theta).unwrap_or_else(|e| panic!("{e}"));
    self.theta = Some(theta);
  }

  fn theta_bounds(&self) -> (f64, f64) {
    (-1.0, 1.0)
  }

  fn invalid_thetas(&self) -> Vec<f64> {
    vec![]
  }

  /// Density $c(u,v) = 1 + \theta(1 - 2u)(1 - 2v)$.
  fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    copula_density(x, |u, v| 1.0 + theta * (1.0 - 2.0 * u) * (1.0 - 2.0 * v))
  }

  /// CDF $C(u,v) = uv + \theta u(1-u) v(1-v)$.
  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    copula_cdf(x, |u, v| u * v + theta * u * (1.0 - u) * v * (1.0 - v))
  }

  /// $\partial_v C(u,v) = u + \theta u (1-u)(1 - 2v)$.
  fn partial_derivative(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    conditional_cdf(x, |u, v| u + theta * u * (1.0 - u) * (1.0 - 2.0 * v))
  }

  /// `Fgm::inverse`'s root.
  fn percent_point_unchecked(
    &self,
    y: ArrayView1<f64>,
    v: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    Ok(conditional_quantiles(
      y,
      v,
      Self::inverse(self.theta.unwrap()),
    ))
  }

  fn tau_from_theta(&self, theta: f64) -> f64 {
    2.0 * theta / 9.0
  }

  fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError> {
    if !(-2.0 / 9.0..=2.0 / 9.0).contains(&tau) {
      return Err(tau_out_of_range(tau, "-2/9 <= tau <= 2/9".into()));
    }
    admitted(self, tau, 4.5 * tau)
  }

  /// FGM is a bounded perturbation of independence — no tail dependence
  /// in either tail, for any $\theta \in [-1, 1]$.
  /// Reference: Nelsen, R.B. (2006), Table 5.1.
  fn tail_dependence(&self) -> Result<TailDependence<f64>, CopulaError> {
    self.check_theta()?;
    Ok(TailDependence {
      lower: 0.0,
      upper: 0.0,
    })
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  #[test]
  fn fgm_independence_theta_zero_reproduces_product_cdf() {
    let mut c = Fgm::default();
    c.set_theta(0.0);
    let x = array![[0.3_f64, 0.7], [0.5, 0.5]];
    let cdf = c.cdf(x.view()).unwrap();
    assert!(approx(cdf[0], 0.21, 1e-12));
    assert!(approx(cdf[1], 0.25, 1e-12));
  }

  #[test]
  fn fgm_pdf_at_center_equals_one_for_zero_theta() {
    let mut c = Fgm::default();
    c.set_theta(0.0);
    let x = array![[0.5_f64, 0.5]];
    let pdf = c.pdf(x.view()).unwrap();
    assert!(approx(pdf[0], 1.0, 1e-12));
  }

  #[test]
  fn fgm_theta_from_tau_inverts_tau_exactly() {
    let c = Fgm::default();
    assert!(approx(c.theta_from_tau(0.1).unwrap(), 0.45, 1e-12));
    assert!(approx(c.theta_from_tau(-0.15).unwrap(), -0.675, 1e-12));
  }

  #[test]
  fn fgm_tau_beyond_two_ninths_is_an_error() {
    let c = Fgm::default();
    for tau in [0.5, -0.5, f64::NAN] {
      assert!(
        matches!(
          c.theta_from_tau(tau),
          Err(CopulaError::InvalidParameter { name: "tau", .. })
        ),
        "τ = {tau}"
      );
    }
  }

  #[test]
  fn fgm_generator_returns_err_not_archimedean() {
    let c = Fgm::default();
    let t = array![0.5_f64, 0.8];
    assert!(c.generator(t.view()).is_err());
  }

  #[test]
  fn fgm_tail_dependence_is_zero() {
    let mut c = Fgm::default();
    c.set_theta(0.9);
    let td = c.tail_dependence().unwrap();
    assert_eq!(td.lower, 0.0);
    assert_eq!(td.upper, 0.0);
  }
}
