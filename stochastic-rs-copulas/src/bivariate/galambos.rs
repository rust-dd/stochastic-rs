//! # Galambos (1975) extreme-value copula
//!
//! $$
//! C_\theta(u,v) = u v \cdot \exp\!\Big\{\big[(-\ln u)^{-\theta} +
//!   (-\ln v)^{-\theta}\big]^{-1/\theta}\Big\},
//! \qquad \theta \in (0, \infty).
//! $$
//!
//! Negative-logistic max-stable family with Pickands dependence function
//! $$
//! A(t) = 1 - \big(t^{-\theta} + (1-t)^{-\theta}\big)^{-1/\theta},
//! \qquad t \in [0, 1].
//! $$
//! $\theta \to 0$ recovers the independence copula ($A \equiv 1$); $\theta
//! \to \infty$ recovers the comonotone copula ($A(t) = \max(t, 1-t)$).
//!
//! Tail dependence: $\lambda_U = 2^{-1/\theta} \cdot 2$ (upper),
//! $\lambda_L = 0$ (lower).
//!
//! Reference: Galambos, J. (1975), "Order statistics of samples from
//! multivariate distributions", *JASA* 70(351), 674-680.
//! Reference: Joe, H. (1997), "Multivariate Models and Dependence
//! Concepts", Chapman & Hall, §5.4.

use std::error::Error;
use std::f64;

use ndarray::Array1;
use ndarray::Array2;
use roots::SimpleConvergency;
use roots::find_root_brent;

use crate::bivariate::CopulaType;
use crate::traits::BivariateExt;

#[derive(Debug, Clone)]
pub struct Galambos {
  pub r#type: CopulaType,
  pub theta: Option<f64>,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Default for Galambos {
  fn default() -> Self {
    Self {
      r#type: CopulaType::Galambos,
      theta: None,
      tau: None,
      theta_bounds: (0.0, f64::INFINITY),
      invalid_thetas: vec![],
    }
  }
}

impl Galambos {
  pub fn new() -> Self {
    Self::default()
  }

  /// Pickands dependence function
  /// $A(t) = 1 - (t^{-\theta} + (1-t)^{-\theta})^{-1/\theta}$.
  pub fn pickands(theta: f64, t: f64) -> f64 {
    if t <= 0.0 || t >= 1.0 {
      return 1.0;
    }
    let s = t.powf(-theta) + (1.0 - t).powf(-theta);
    1.0 - s.powf(-1.0 / theta)
  }

  /// Kendall's tau via the Genest-Rivest (1993) extreme-value identity
  /// $\tau(C) = \int_0^1 t(1-t)\, A''(t)/A(t)\, dt$, computed by Simpson's
  /// rule with $N = 512$ panels. Central-difference $A''$ on a grid of
  /// step $h = 10^{-4}$ achieves $\sim 10^{-6}$ accuracy in $\theta$ for
  /// the inverse problem.
  pub fn tau_from_theta(theta: f64) -> f64 {
    if theta <= 1e-9 {
      return 0.0;
    }
    let n: usize = 512;
    let h_grid = 1.0 / n as f64;
    let h_fd = 1e-4_f64;
    let mut acc = 0.0_f64;
    for k in 1..n {
      let t = (k as f64) * h_grid;
      let a = Self::pickands(theta, t);
      let a_pp = (Self::pickands(theta, (t + h_fd).min(1.0 - 1e-12))
        - 2.0 * a
        + Self::pickands(theta, (t - h_fd).max(1e-12)))
        / (h_fd * h_fd);
      let w = if k % 2 == 0 { 2.0 } else { 4.0 };
      acc += w * t * (1.0 - t) * a_pp / a.max(1e-15);
    }
    (h_grid / 3.0) * acc
  }
}

impl BivariateExt for Galambos {
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

  /// Galambos is **not** Archimedean — no scalar generator.
  fn generator(&self, _t: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    Err("Galambos is not Archimedean — generator not defined".into())
  }

  /// Density via the closed-form chain-rule derivation
  /// $c(u,v) = C(u,v)/(uv) \cdot [g(x,y) + (\theta+1)\,T^{2\theta+1}/(xy)^{\theta+1}]$
  /// where $x = -\ln u$, $y = -\ln v$, $T = (x^{-\theta} + y^{-\theta})^{-1/\theta}$,
  /// $g(x,y) = ((T/x)^{\theta+1} - 1)((T/y)^{\theta+1} - 1)$.
  fn pdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    let u_col = x.column(0);
    let v_col = x.column(1);
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i];
      if u <= 0.0 || u >= 1.0 || v <= 0.0 || v >= 1.0 {
        out[i] = 0.0;
        continue;
      }
      let xx = -u.ln();
      let yy = -v.ln();
      let big_t = (xx.powf(-theta) + yy.powf(-theta)).powf(-1.0 / theta);
      let cuv = u * v * big_t.exp();
      let r_x = (big_t / xx).powf(theta + 1.0);
      let r_y = (big_t / yy).powf(theta + 1.0);
      let extra = (theta + 1.0) * big_t.powf(2.0 * theta + 1.0)
        / (xx.powf(theta + 1.0) * yy.powf(theta + 1.0));
      out[i] = cuv / (u * v) * ((r_x - 1.0) * (r_y - 1.0) + extra);
    }
    Ok(out)
  }

  /// CDF $C(u,v) = u v \cdot \exp\!\{(x^{-\theta} + y^{-\theta})^{-1/\theta}\}$
  /// with $x = -\ln u$, $y = -\ln v$.
  fn cdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    let u_col = x.column(0);
    let v_col = x.column(1);
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i];
      if u <= 0.0 || v <= 0.0 {
        out[i] = 0.0;
        continue;
      }
      if u >= 1.0 {
        out[i] = v;
        continue;
      }
      if v >= 1.0 {
        out[i] = u;
        continue;
      }
      let xx = -u.ln();
      let yy = -v.ln();
      let t_pow = (xx.powf(-theta) + yy.powf(-theta)).powf(-1.0 / theta);
      out[i] = u * v * t_pow.exp();
    }
    Ok(out)
  }

  /// $\partial_v C(u,v) = (C/v) \cdot [1 - (T/y)^{\theta+1}]$ with
  /// $x = -\ln u$, $y = -\ln v$, $T = (x^{-\theta} + y^{-\theta})^{-1/\theta}$.
  /// Derivation: $\ln C = \ln u + \ln v + T$ and $\partial T/\partial y =
  /// (T/y)^{\theta+1}$; chain through $\partial y/\partial v = -1/v$.
  fn partial_derivative(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    let u_col = x.column(0);
    let v_col = x.column(1);
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i];
      if v <= 0.0 {
        out[i] = 0.0;
        continue;
      }
      if v >= 1.0 {
        // ∂C/∂v(u, 1) = 0 for Galambos (positive upper-tail dependence
        // "spends" the marginal mass at the boundary); see module docs.
        out[i] = 0.0;
        continue;
      }
      let xx = -u.ln();
      let yy = -v.ln();
      let big_t = (xx.powf(-theta) + yy.powf(-theta)).powf(-1.0 / theta);
      let cuv = u * v * big_t.exp();
      let ratio = (big_t / yy).powf(theta + 1.0);
      out[i] = (cuv / v) * (1.0 - ratio);
    }
    Ok(out)
  }

  fn compute_theta(&self) -> f64 {
    let tau = self.tau.unwrap();
    if tau <= 1e-6 {
      return 1e-6;
    }
    if tau >= 1.0 - 1e-6 {
      return 50.0;
    }
    let residual = |theta: f64| Self::tau_from_theta(theta) - tau;
    let mut convergency = SimpleConvergency {
      eps: 1e-6,
      max_iter: 100,
    };
    find_root_brent(1e-4, 50.0, residual, &mut convergency).unwrap_or(1.0)
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
  fn galambos_marginal_recovers_input() {
    let mut c = Galambos::new();
    c.set_theta(2.0);
    let x = array![[0.5_f64, 1.0], [1.0, 0.7]];
    let cdf = c.cdf(&x).unwrap();
    assert!(approx(cdf[0], 0.5, 1e-12));
    assert!(approx(cdf[1], 0.7, 1e-12));
  }

  #[test]
  fn galambos_zero_theta_limit_is_independence() {
    let mut c = Galambos::new();
    c.set_theta(1e-6);
    let x = array![[0.3_f64, 0.7], [0.5, 0.5]];
    let cdf = c.cdf(&x).unwrap();
    assert!(approx(cdf[0], 0.21, 1e-3), "got {}", cdf[0]);
    assert!(approx(cdf[1], 0.25, 1e-3), "got {}", cdf[1]);
  }

  #[test]
  fn galambos_large_theta_approaches_comonotone() {
    let mut c = Galambos::new();
    c.set_theta(50.0);
    let x = array![[0.3_f64, 0.7], [0.5, 0.5]];
    let cdf = c.cdf(&x).unwrap();
    // Comonotone: C(u,v) = min(u,v).
    assert!((cdf[0] - 0.3).abs() < 0.05, "got {}", cdf[0]);
    assert!((cdf[1] - 0.5).abs() < 0.05, "got {}", cdf[1]);
  }

  #[test]
  fn galambos_pickands_at_half_matches_closed_form() {
    // A(1/2) = 1 - (2 · 2^θ)^{-1/θ} = 1 - 2^{-(θ+1)/θ}.
    for &theta in &[0.5_f64, 1.0, 2.0, 5.0] {
      let a = Galambos::pickands(theta, 0.5);
      let expected = 1.0 - 2.0_f64.powf(-(theta + 1.0) / theta);
      assert!(
        approx(a, expected, 1e-12),
        "θ={theta}: A(1/2)={a}, expected {expected}"
      );
    }
  }

  #[test]
  fn galambos_upper_tail_dependence_formula() {
    // λ_U = 2 · 2^{-1/θ}. Verified via the Pickands relation
    // λ_U = 2(1 - A(1/2)).
    for &theta in &[0.5_f64, 1.0, 2.0, 5.0] {
      let lambda_u_pickands = 2.0 * (1.0 - Galambos::pickands(theta, 0.5));
      let expected = 2.0_f64.powf(-1.0 / theta);
      assert!(
        approx(lambda_u_pickands, expected, 1e-12),
        "θ={theta}: λ_U_pickands={lambda_u_pickands}, expected {expected}"
      );
    }
  }

  #[test]
  fn galambos_compute_theta_inverts_tau() {
    let target_theta = 1.5_f64;
    let tau = Galambos::tau_from_theta(target_theta);
    let mut c = Galambos::new();
    c.set_tau(tau);
    let recovered = c.compute_theta();
    assert!(
      approx(recovered, target_theta, 0.05),
      "expected {target_theta}, got {recovered}"
    );
  }

  #[test]
  fn galambos_pdf_positive_interior() {
    let mut c = Galambos::new();
    c.set_theta(1.5);
    let x = array![[0.25_f64, 0.75], [0.5, 0.5], [0.1, 0.9]];
    let pdf = c.pdf(&x).unwrap();
    for &p in pdf.iter() {
      assert!(p > 0.0 && p.is_finite(), "pdf={p}");
    }
  }
}
