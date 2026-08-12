//! # Hüsler-Reiss (1989) extreme-value copula
//!
//! ```text
//! C_λ(u,v) = exp{ ln u · Φ(1/λ + (λ/2)·ln(ln u / ln v))
//!                + ln v · Φ(1/λ + (λ/2)·ln(ln v / ln u)) },   λ ∈ (0, ∞).
//! ```
//!
//! Gaussian-limit max-stable family with Pickands dependence function
//!
//! ```text
//! A(t) = (1-t)·Φ(1/λ + (λ/2)·ln((1-t)/t)) + t·Φ(1/λ + (λ/2)·ln(t/(1-t))),
//!        t ∈ [0, 1].
//! ```
//!
//! $\lambda \to 0$ recovers the independence copula
//! ($\Phi(\pm\infty) = \{0, 1\}$ collapse $A$ to $1$);
//! $\lambda \to \infty$ recovers the comonotone copula.
//!
//! Tail dependence: $\lambda_U = 2(1 - \Phi(1/\lambda))$ (upper),
//! $\lambda_L = 0$ (lower).
//!
//! The implementation exploits the Hüsler-Reiss identity
//! $x\,\varphi(\alpha) = y\,\varphi(\beta)$ for the symmetric arguments
//! $\alpha = 1/\lambda + (\lambda/2)\ln(x/y)$,
//! $\beta = 2/\lambda - \alpha$, which collapses the otherwise messy
//! derivatives of $\ln C$ to $\partial_y \ln C = -\Phi(\beta)$.
//!
//! Reference: Hüsler, J., Reiss, R.-D. (1989), "Maxima of normal random
//! vectors: between independence and complete dependence", *Statist.
//! Probab. Lett.* 7(4), 283-286.

use std::error::Error;
use std::f64;

use ndarray::Array1;
use ndarray::Array2;
use roots::SimpleConvergency;
use roots::find_root_brent;
use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_distributions::special::norm_pdf;

use crate::bivariate::CopulaType;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;

/// Clamp used to evaluate [`HuslerReiss::partial_derivative`] just inside
/// `(0, 1)` at the `v` boundary instead of returning a hardcoded constant
/// there — see that method's doc for why a constant is wrong.
const BOUNDARY_EPS: f64 = 1e-12;

#[derive(Debug, Clone)]
pub struct HuslerReiss {
  pub r#type: CopulaType,
  /// Hüsler-Reiss parameter $\lambda > 0$ (stored in the `theta` field for
  /// trait compatibility with single-parameter Archimedean copulas).
  pub theta: Option<f64>,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Default for HuslerReiss {
  fn default() -> Self {
    Self {
      r#type: CopulaType::HuslerReiss,
      theta: None,
      tau: None,
      theta_bounds: (0.0, f64::INFINITY),
      invalid_thetas: vec![],
    }
  }
}

impl HuslerReiss {
  pub fn new() -> Self {
    Self::default()
  }

  /// Pickands dependence function.
  pub fn pickands(lambda: f64, t: f64) -> f64 {
    if t <= 0.0 {
      return 1.0;
    }
    if t >= 1.0 {
      return 1.0;
    }
    let log_ratio = ((1.0 - t) / t).ln();
    let alpha = 1.0 / lambda + 0.5 * lambda * log_ratio;
    let beta = 1.0 / lambda - 0.5 * lambda * log_ratio;
    (1.0 - t) * norm_cdf(alpha) + t * norm_cdf(beta)
  }

  /// Kendall's tau via the Genest-Rivest extreme-value identity, Simpson
  /// integration with $N = 512$ panels and central-difference $A''$ on a
  /// step of $h = 10^{-4}$.
  pub fn tau_from_lambda(lambda: f64) -> f64 {
    if lambda <= 1e-9 {
      return 0.0;
    }
    let n: usize = 512;
    let h_grid = 1.0 / n as f64;
    let h_fd = 1e-4_f64;
    let mut acc = 0.0_f64;
    for k in 1..n {
      let t = (k as f64) * h_grid;
      let a = Self::pickands(lambda, t);
      let a_pp = (Self::pickands(lambda, (t + h_fd).min(1.0 - 1e-12)) - 2.0 * a
        + Self::pickands(lambda, (t - h_fd).max(1e-12)))
        / (h_fd * h_fd);
      let w = if k % 2 == 0 { 2.0 } else { 4.0 };
      acc += w * t * (1.0 - t) * a_pp / a.max(1e-15);
    }
    (h_grid / 3.0) * acc
  }
}

impl BivariateExt for HuslerReiss {
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

  /// Density $c(u,v) = C(u,v) / (uv) \cdot [\Phi(\alpha)\Phi(\beta) +
  /// (\lambda/2)\varphi(\alpha)/y]$ with $x = -\ln u$, $y = -\ln v$,
  /// $\alpha = 1/\lambda + (\lambda/2)\ln(x/y)$, $\beta = 2/\lambda -
  /// \alpha$. Uses the Hüsler-Reiss identity for stability.
  fn pdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;
    let lambda = self.theta.unwrap();
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
      let half_log = 0.5 * (xx / yy).ln();
      let alpha = 1.0 / lambda + lambda * half_log;
      let beta = 1.0 / lambda - lambda * half_log;
      let cuv = (-xx * norm_cdf(alpha) - yy * norm_cdf(beta)).exp();
      let extra = 0.5 * lambda * norm_pdf(alpha) / yy;
      out[i] = cuv / (u * v) * (norm_cdf(alpha) * norm_cdf(beta) + extra);
    }
    Ok(out)
  }

  fn cdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;
    let lambda = self.theta.unwrap();
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
      let half_log = 0.5 * (xx / yy).ln();
      let alpha = 1.0 / lambda + lambda * half_log;
      let beta = 1.0 / lambda - lambda * half_log;
      out[i] = (-xx * norm_cdf(alpha) - yy * norm_cdf(beta)).exp();
    }
    Ok(out)
  }

  /// $\partial_v C(u,v) = C(u,v) \cdot \Phi(\beta) / v$.
  ///
  /// At the `v \to 0^+/1^-` boundary, `y = -\ln v \to \pm\infty` drives
  /// `\alpha,\beta` (and hence `C(u,v)\cdot\Phi(\beta)/v`) through both an
  /// exponential prefactor and a `\Phi`-of-log-ratio term at once, and the
  /// combined limit is not a family-wide constant: numerically it depends
  /// on both `u` and `\lambda` at `v\to0^+` (e.g. `\lambda=0.2` limits near
  /// `u`, `\lambda\ge1` limits near `1`), so `0.0` is wrong whenever
  /// `\lambda` is not small. At `v\to1^-` the limit does converge to `0`
  /// for every `\lambda` tested, but not to the hardcoded `1.0` this branch
  /// returned. Since no family-wide closed form falls out cleanly (unlike
  /// [`crate::bivariate::t_copula::TCopula::partial_derivative`]), evaluate
  /// the real formula at `v` clamped just inside `(0,1)` for both
  /// boundaries instead — continuous in `v`, so it tracks the true
  /// one-sided limit.
  fn partial_derivative(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;
    let lambda = self.theta.unwrap();
    let u_col = x.column(0);
    let v_col = x.column(1);
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i].clamp(BOUNDARY_EPS, 1.0 - BOUNDARY_EPS);
      let xx = -u.ln();
      let yy = -v.ln();
      let half_log = 0.5 * (xx / yy).ln();
      let alpha = 1.0 / lambda + lambda * half_log;
      let beta = 1.0 / lambda - lambda * half_log;
      let cuv = (-xx * norm_cdf(alpha) - yy * norm_cdf(beta)).exp();
      out[i] = cuv * norm_cdf(beta) / v;
    }
    Ok(out)
  }

  fn compute_theta(&self) -> f64 {
    let tau = self.tau.unwrap();
    if tau <= 1e-6 {
      return 1e-6;
    }
    if tau >= 1.0 - 1e-6 {
      return 20.0;
    }
    let residual = |lambda: f64| Self::tau_from_lambda(lambda) - tau;
    let mut convergency = SimpleConvergency {
      eps: 1e-6,
      max_iter: 100,
    };
    find_root_brent(1e-3, 20.0, residual, &mut convergency).unwrap_or(1.0)
  }

  /// Upper-tail dependence $\lambda_U = 2(1 - \Phi(1/\lambda))$;
  /// Hüsler-Reiss has no lower-tail dependence.
  /// Reference: Hüsler, J., Reiss, R.-D. (1989) (module header).
  fn tail_dependence(&self) -> TailDependence<f64> {
    self.assert_theta_valid_for_tail_dependence();
    let lambda = self.theta.unwrap();
    TailDependence {
      lower: 0.0,
      upper: 2.0 * (1.0 - norm_cdf(1.0 / lambda)),
    }
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
  fn hr_marginal_recovers_input() {
    let mut c = HuslerReiss::new();
    c.set_theta(1.5);
    let x = array![[0.5_f64, 1.0], [1.0, 0.7]];
    let cdf = c.cdf(&x).unwrap();
    assert!(approx(cdf[0], 0.5, 1e-12));
    assert!(approx(cdf[1], 0.7, 1e-12));
  }

  #[test]
  fn hr_small_lambda_limit_is_independence() {
    let mut c = HuslerReiss::new();
    c.set_theta(1e-3);
    let x = array![[0.3_f64, 0.7], [0.5, 0.5]];
    let cdf = c.cdf(&x).unwrap();
    assert!(approx(cdf[0], 0.21, 1e-3), "got {}", cdf[0]);
    assert!(approx(cdf[1], 0.25, 1e-3), "got {}", cdf[1]);
  }

  #[test]
  fn hr_large_lambda_approaches_comonotone() {
    let mut c = HuslerReiss::new();
    c.set_theta(15.0);
    let x = array![[0.3_f64, 0.7], [0.5, 0.5]];
    let cdf = c.cdf(&x).unwrap();
    assert!((cdf[0] - 0.3).abs() < 0.05, "got {}", cdf[0]);
    assert!((cdf[1] - 0.5).abs() < 0.05, "got {}", cdf[1]);
  }

  #[test]
  fn hr_upper_tail_dependence_formula() {
    // λ_U = 2(1 - Φ(1/λ)). Verified via the Pickands relation
    // λ_U = 2(1 - A(1/2)).
    for &lambda in &[0.5_f64, 1.0, 2.0, 5.0] {
      let lambda_u_pickands = 2.0 * (1.0 - HuslerReiss::pickands(lambda, 0.5));
      let expected = 2.0 * (1.0 - norm_cdf(1.0 / lambda));
      assert!(
        approx(lambda_u_pickands, expected, 1e-12),
        "λ={lambda}: λ_U_pickands={lambda_u_pickands}, expected {expected}"
      );
    }
  }

  #[test]
  fn hr_pickands_symmetry_at_one_half() {
    // At t = 1/2, log_ratio = 0, so α = β = 1/λ; A(1/2) = Φ(1/λ).
    for &lambda in &[0.5_f64, 1.0, 2.0, 5.0] {
      let a = HuslerReiss::pickands(lambda, 0.5);
      let expected = norm_cdf(1.0 / lambda);
      assert!(
        approx(a, expected, 1e-12),
        "λ={lambda}: A(1/2)={a}, expected {expected}"
      );
    }
  }

  #[test]
  fn hr_compute_theta_inverts_tau() {
    let target_lambda = 1.5_f64;
    let tau = HuslerReiss::tau_from_lambda(target_lambda);
    let mut c = HuslerReiss::new();
    c.set_tau(tau);
    let recovered = c.compute_theta();
    assert!(
      approx(recovered, target_lambda, 0.05),
      "expected {target_lambda}, got {recovered}"
    );
  }

  #[test]
  fn hr_pdf_positive_interior() {
    let mut c = HuslerReiss::new();
    c.set_theta(1.5);
    let x = array![[0.25_f64, 0.75], [0.5, 0.5], [0.1, 0.9]];
    let pdf = c.pdf(&x).unwrap();
    for &p in pdf.iter() {
      assert!(p > 0.0 && p.is_finite(), "pdf={p}");
    }
  }

  #[test]
  fn hr_tail_dependence_matches_formula() {
    let mut c = HuslerReiss::new();
    c.set_theta(1.5);
    let td = c.tail_dependence();
    let expected = 2.0 * (1.0 - norm_cdf(1.0 / 1.5));
    assert!(approx(td.upper, expected, 1e-12), "got {}", td.upper);
    assert_eq!(td.lower, 0.0);
  }

  /// `partial_derivative` must not jump to a hardcoded constant right at
  /// the `v` boundary: its value at `v=1e-12` should be continuous with
  /// its value at `v=1e-9` (the F8 regression this guards). Checked at
  /// `\lambda=1.5`, where the pre-fix code hardcoded `0.0` near `v=0` even
  /// though the true limit there is close to `1` for this `\lambda`, and
  /// hardcoded `1.0` near `v=1` even though the true limit there is `0`
  /// (see the method's doc).
  ///
  /// Tolerance `2e-2`, not `1e-6`: unlike the Gaussian/t-copula families,
  /// this family's `v\to0^+` convergence is not fast enough for a `1e-6`
  /// bound to hold across a `1000\times` shrink in `v` — measured directly
  /// against this formula at `\lambda=1.5,u=0.4`, the real gap between
  /// `v=1e-12` and `v=1e-9` is `\approx 7.0e-3`. `2e-2` still rejects a
  /// jump to a hardcoded constant by more than an order of magnitude (such
  /// a jump is `\approx 1.0` here, since the true limit sits near `1`).
  #[test]
  fn hr_partial_derivative_no_jump_near_v_boundary() {
    let mut c = HuslerReiss::new();
    c.set_theta(1.5);
    let u = 0.4_f64;
    let near_zero_a = c.partial_derivative(&array![[u, 1e-12]]).unwrap()[0];
    let near_zero_b = c.partial_derivative(&array![[u, 1e-9]]).unwrap()[0];
    assert!(
      approx(near_zero_a, near_zero_b, 2e-2),
      "expected continuity near v=0: v=1e-12 -> {near_zero_a}, v=1e-9 -> {near_zero_b}"
    );
    let near_one_a = c.partial_derivative(&array![[u, 1.0 - 1e-12]]).unwrap()[0];
    let near_one_b = c.partial_derivative(&array![[u, 1.0 - 1e-9]]).unwrap()[0];
    assert!(
      approx(near_one_a, near_one_b, 2e-2),
      "expected continuity near v=1: v=1-1e-12 -> {near_one_a}, v=1-1e-9 -> {near_one_b}"
    );

    // Directional regression check: at lambda=1.5 the true v->0+ limit is
    // near 1 (not the pre-fix hardcoded 0.0), and the true v->1- limit is
    // near 0 (not the pre-fix hardcoded 1.0).
    assert!(
      near_zero_a > 0.9,
      "lambda=1.5 v->0+ limit should be close to 1, got {near_zero_a}"
    );
    assert!(
      near_one_a < 0.1,
      "lambda=1.5 v->1- limit should be close to 0, got {near_one_a}"
    );
  }

  /// `generator` has no override in this family, so this exercises
  /// `BivariateExt::generator`'s trait-default body directly.
  #[test]
  fn hr_generator_returns_err_not_archimedean() {
    let c = HuslerReiss::new();
    let t = array![0.5_f64, 0.8];
    assert!(c.generator(&t).is_err());
  }
}
