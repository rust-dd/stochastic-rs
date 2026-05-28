//! # Marshall-Olkin (1967) bivariate copula
//!
//! $$
//! C_{\alpha,\beta}(u,v) =
//! \begin{cases}
//!   u^{1-\alpha}\, v & \text{if } u^\alpha \ge v^\beta,\\
//!   u\, v^{1-\beta} & \text{if } u^\alpha < v^\beta,
//! \end{cases}
//! \qquad \alpha, \beta \in (0, 1].
//! $$
//!
//! Equivalently $C(u,v) = \min\!\big(u^{1-\alpha} v,\, u v^{1-\beta}\big)$.
//! Non-exchangeable in general; **has a singular component** on the curve
//! $u^\alpha = v^\beta$ carrying mass $\alpha\beta / (\alpha + \beta -
//! \alpha\beta)$. The remaining $1 - \alpha\beta/(\alpha+\beta-\alpha\beta)$
//! of probability mass is absolutely continuous on the two open sectors.
//!
//! Tail dependence: $\lambda_U = \min(\alpha, \beta)$ (upper),
//! $\lambda_L = 0$ (lower).
//!
//! Kendall's tau:
//! $\tau = \alpha\beta / (\alpha + \beta - \alpha\beta)$.
//!
//! [`BivariateExt::set_theta`] / [`BivariateExt::compute_theta`] operate on
//! the *symmetric* slice $\alpha = \beta = \theta$, in which case
//! $\tau = \theta / (2 - \theta) \iff \theta = 2\tau/(1 + \tau)$. The
//! asymmetric two-parameter case is supplied through
//! [`MarshallOlkin::with_alpha_beta`].
//!
//! Reference: Marshall, A.W., Olkin, I. (1967), "A multivariate exponential
//! distribution", *JASA* 62(317), 30-44.
//! Reference: Nelsen, R.B. (2006), "An Introduction to Copulas", 2nd ed.,
//! Springer, Example 3.6.

use std::error::Error;
use std::f64;

use ndarray::Array1;
use ndarray::Array2;

use crate::bivariate::CopulaType;
use crate::traits::BivariateExt;

#[derive(Debug, Clone)]
pub struct MarshallOlkin {
  pub r#type: CopulaType,
  pub theta: Option<f64>,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
  /// Asymmetric Marshall-Olkin parameter $\alpha \in (0, 1]$. When
  /// `Some`, takes precedence over the symmetric `theta` field.
  pub alpha: Option<f64>,
  /// Asymmetric Marshall-Olkin parameter $\beta \in (0, 1]$.
  pub beta: Option<f64>,
}

impl Default for MarshallOlkin {
  fn default() -> Self {
    Self {
      r#type: CopulaType::MarshallOlkin,
      theta: None,
      tau: None,
      theta_bounds: (0.0, 1.0),
      invalid_thetas: vec![],
      alpha: None,
      beta: None,
    }
  }
}

impl MarshallOlkin {
  pub fn new() -> Self {
    Self::default()
  }

  /// Construct the asymmetric two-parameter variant $C_{\alpha,\beta}$.
  /// Both parameters must lie in $(0, 1]$.
  pub fn with_alpha_beta(alpha: f64, beta: f64) -> Self {
    assert!(
      alpha > 0.0 && alpha <= 1.0,
      "alpha must lie in (0, 1], got {alpha}"
    );
    assert!(
      beta > 0.0 && beta <= 1.0,
      "beta must lie in (0, 1], got {beta}"
    );
    Self {
      alpha: Some(alpha),
      beta: Some(beta),
      ..Self::default()
    }
  }

  /// Resolve the effective `(alpha, beta)` pair from either the asymmetric
  /// fields or the symmetric `theta` fallback.
  fn resolve_params(&self) -> (f64, f64) {
    match (self.alpha, self.beta) {
      (Some(a), Some(b)) => (a, b),
      _ => {
        let theta = self
          .theta
          .expect("Marshall-Olkin: neither (alpha, beta) nor theta is set");
        (theta, theta)
      }
    }
  }
}

impl BivariateExt for MarshallOlkin {
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
    // Reset the asymmetric slot so the symmetric value wins on resolve.
    self.alpha = None;
    self.beta = None;
  }

  /// Marshall-Olkin is **not** Archimedean — no generator.
  fn generator(&self, _t: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    Err("Marshall-Olkin is not Archimedean — generator not defined".into())
  }

  /// Absolutely continuous density. Returns `0` exactly on the singular
  /// curve $u^\alpha = v^\beta$ and `(1 - \alpha) u^{-\alpha}` /
  /// `(1 - \beta) v^{-\beta}` in the two open sectors.
  fn pdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    let (alpha, beta) = self.resolve_params();
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
      let lhs = u.powf(alpha);
      let rhs = v.powf(beta);
      if (lhs - rhs).abs() < 1e-15 {
        // Singular curve: absolutely continuous density is 0; the singular
        // component carries the curve mass and is not returned here.
        out[i] = 0.0;
      } else if lhs > rhs {
        out[i] = (1.0 - alpha) * u.powf(-alpha);
      } else {
        out[i] = (1.0 - beta) * v.powf(-beta);
      }
    }
    Ok(out)
  }

  /// CDF $C_{\alpha,\beta}(u,v) = \min(u^{1-\alpha} v, u v^{1-\beta})$.
  fn cdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    let (alpha, beta) = self.resolve_params();
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
      let lhs = u.powf(alpha);
      let rhs = v.powf(beta);
      out[i] = if lhs >= rhs {
        u.powf(1.0 - alpha) * v
      } else {
        u * v.powf(1.0 - beta)
      };
    }
    Ok(out)
  }

  /// $\partial_u C$. Continuous everywhere except across the singular
  /// curve (where it jumps).
  fn partial_derivative(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    let (alpha, beta) = self.resolve_params();
    let u_col = x.column(0);
    let v_col = x.column(1);
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i];
      if u <= 0.0 {
        out[i] = 0.0;
        continue;
      }
      if u >= 1.0 {
        out[i] = 1.0;
        continue;
      }
      let lhs = u.powf(alpha);
      let rhs = v.powf(beta);
      out[i] = if lhs >= rhs {
        (1.0 - alpha) * u.powf(-alpha) * v
      } else {
        v.powf(1.0 - beta)
      };
    }
    Ok(out)
  }

  /// Symmetric-slice Kendall's tau inversion: $\theta = 2\tau / (1 + \tau)$.
  /// For the asymmetric case, $\tau$ alone underdetermines $(\alpha, \beta)$ —
  /// supply both directly via [`MarshallOlkin::with_alpha_beta`].
  fn compute_theta(&self) -> f64 {
    let tau = self.tau.unwrap();
    if tau <= 0.0 {
      return 0.0_f64.max(f64::EPSILON);
    }
    if tau >= 1.0 - 1e-12 {
      return 1.0;
    }
    (2.0 * tau / (1.0 + tau)).clamp(0.0, 1.0)
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
  fn mo_cdf_marginal_recovers_input() {
    let c = MarshallOlkin::with_alpha_beta(0.5, 0.3);
    let x = array![[0.4_f64, 1.0], [1.0, 0.7]];
    let cdf = c.cdf(&x).unwrap();
    assert!(approx(cdf[0], 0.4, 1e-12));
    assert!(approx(cdf[1], 0.7, 1e-12));
  }

  #[test]
  fn mo_alpha_eq_one_beta_eq_one_is_comonotone() {
    let c = MarshallOlkin::with_alpha_beta(1.0, 1.0);
    let x = array![[0.3_f64, 0.7], [0.6, 0.2], [0.5, 0.5]];
    let cdf = c.cdf(&x).unwrap();
    for i in 0..x.nrows() {
      let expected = x[[i, 0]].min(x[[i, 1]]);
      assert!(approx(cdf[i], expected, 1e-12), "row {i}");
    }
  }

  #[test]
  fn mo_alpha_zero_or_beta_zero_is_independence() {
    // α → 0 with β fixed: C(u,v) = u^{1-0} v = u v. Use α just above 0.
    let c = MarshallOlkin::with_alpha_beta(1e-12, 0.5);
    let x = array![[0.4_f64, 0.6]];
    let cdf = c.cdf(&x).unwrap();
    assert!(approx(cdf[0], 0.24, 1e-6), "α→0: got {}", cdf[0]);
  }

  #[test]
  fn mo_compute_theta_via_symmetric_inversion() {
    // Symmetric MO: τ = θ/(2-θ). Pick θ = 0.5 ⇒ τ = 1/3; invert to recover.
    let mut c = MarshallOlkin::new();
    c.set_tau(1.0 / 3.0);
    let theta = c.compute_theta();
    assert!(approx(theta, 0.5, 1e-12), "expected θ=0.5, got {theta}");
  }

  #[test]
  fn mo_singular_curve_total_mass_matches_paper() {
    // Singular component carries mass αβ/(α+β-αβ). Verify against
    // Monte-Carlo on a fine grid: count fraction of unit-square sectors
    // dominated by the absolutely continuous density vs total.
    let alpha = 0.6_f64;
    let beta = 0.4_f64;
    let mass_singular_paper = alpha * beta / (alpha + beta - alpha * beta);

    // Integrate the absolutely continuous density on a 200×200 grid.
    let c = MarshallOlkin::with_alpha_beta(alpha, beta);
    let n = 200usize;
    let h = 1.0 / n as f64;
    let mut points = Array2::<f64>::zeros((n * n, 2));
    for i in 0..n {
      for j in 0..n {
        let row = i * n + j;
        points[[row, 0]] = (i as f64 + 0.5) * h;
        points[[row, 1]] = (j as f64 + 0.5) * h;
      }
    }
    let pdf_vals = c.pdf(&points).unwrap();
    let mass_abs: f64 = pdf_vals.iter().sum::<f64>() * h * h;
    let mass_singular_grid = 1.0 - mass_abs;

    assert!(
      (mass_singular_grid - mass_singular_paper).abs() < 0.02,
      "singular mass grid={mass_singular_grid:.4}, paper={mass_singular_paper:.4}"
    );
  }

  #[test]
  fn mo_partial_derivative_in_each_sector() {
    let c = MarshallOlkin::with_alpha_beta(0.5, 0.5);
    // At (u,v) = (0.9, 0.5): u^0.5 = 0.948, v^0.5 = 0.707, lhs > rhs ⇒
    // sector R₁, ∂_u C = (1-α) u^{-α} v = 0.5 · 0.9^{-0.5} · 0.5
    //                  = 0.5 · 1.0541 · 0.5 = 0.2635
    let x = array![[0.9_f64, 0.5]];
    let pd = c.partial_derivative(&x).unwrap();
    let expected = 0.5_f64 * 0.9_f64.powf(-0.5) * 0.5;
    assert!(
      approx(pd[0], expected, 1e-12),
      "pd={}, expected {expected}",
      pd[0]
    );
  }
}
