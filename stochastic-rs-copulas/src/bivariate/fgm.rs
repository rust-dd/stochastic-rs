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

use std::error::Error;
use std::f64;

use ndarray::Array1;
use ndarray::Array2;

use crate::bivariate::CopulaType;
use crate::traits::BivariateExt;

#[derive(Debug, Clone)]
pub struct Fgm {
  pub r#type: CopulaType,
  pub theta: Option<f64>,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Default for Fgm {
  fn default() -> Self {
    Self {
      r#type: CopulaType::Fgm,
      theta: None,
      tau: None,
      theta_bounds: (-1.0, 1.0),
      invalid_thetas: vec![],
    }
  }
}

impl Fgm {
  pub fn new() -> Self {
    Self::default()
  }
}

impl BivariateExt for Fgm {
  fn r#type(&self) -> CopulaType {
    self.r#type
  }

  fn tau(&self) -> Option<f64> {
    self.tau
  }

  fn set_tau(&mut self, tau: f64) {
    self.tau = Some(tau);
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

  /// FGM is **not** an Archimedean copula; the generator is undefined.
  fn generator(&self, _t: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    Err("FGM is not Archimedean — generator not defined".into())
  }

  /// Density $c(u,v) = 1 + \theta(1 - 2u)(1 - 2v)$.
  fn pdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;
    let u_col = x.column(0);
    let v_col = x.column(1);
    let theta = self.theta.unwrap();
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i];
      out[i] = 1.0 + theta * (1.0 - 2.0 * u) * (1.0 - 2.0 * v);
    }
    Ok(out)
  }

  /// CDF $C(u,v) = uv + \theta u(1-u) v(1-v)$.
  fn cdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;
    let u_col = x.column(0);
    let v_col = x.column(1);
    let theta = self.theta.unwrap();
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i];
      out[i] = u * v + theta * u * (1.0 - u) * v * (1.0 - v);
    }
    Ok(out)
  }

  /// $\partial_v C(u,v) = u + \theta u (1-u)(1 - 2v)$.
  fn partial_derivative(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;
    let u_col = x.column(0);
    let v_col = x.column(1);
    let theta = self.theta.unwrap();
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i];
      out[i] = u + theta * u * (1.0 - u) * (1.0 - 2.0 * v);
    }
    Ok(out)
  }

  /// Closed-form Kendall's tau inversion: $\tau = 2\theta/9 \implies \theta = 9\tau/2$.
  /// Saturates at $\pm 1$ when the requested $|\tau| > 2/9$ (the FGM limit).
  fn compute_theta(&self) -> f64 {
    let tau = self.tau.unwrap();
    (9.0 * tau / 2.0).clamp(-1.0, 1.0)
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
    let mut c = Fgm::new();
    c.set_theta(0.0);
    let x = array![[0.3_f64, 0.7], [0.5, 0.5]];
    let cdf = c.cdf(&x).unwrap();
    assert!(approx(cdf[0], 0.21, 1e-12));
    assert!(approx(cdf[1], 0.25, 1e-12));
  }

  #[test]
  fn fgm_pdf_at_center_equals_one_for_zero_theta() {
    let mut c = Fgm::new();
    c.set_theta(0.0);
    let x = array![[0.5_f64, 0.5]];
    let pdf = c.pdf(&x).unwrap();
    assert!(approx(pdf[0], 1.0, 1e-12));
  }

  #[test]
  fn fgm_compute_theta_inverts_tau_exactly() {
    let mut c = Fgm::new();
    c.set_tau(0.1);
    assert!(approx(c.compute_theta(), 0.45, 1e-12));
    c.set_tau(-0.15);
    assert!(approx(c.compute_theta(), -0.675, 1e-12));
  }

  #[test]
  fn fgm_compute_theta_saturates_at_extreme_tau() {
    let mut c = Fgm::new();
    c.set_tau(0.5);
    // 9*0.5/2 = 2.25 → clamped to 1.0 (FGM max correlation).
    assert!(approx(c.compute_theta(), 1.0, 1e-12));
    c.set_tau(-0.5);
    assert!(approx(c.compute_theta(), -1.0, 1e-12));
  }

  #[test]
  fn fgm_generator_returns_err_not_archimedean() {
    let c = Fgm::new();
    let t = array![0.5_f64, 0.8];
    assert!(c.generator(&t).is_err());
  }
}
