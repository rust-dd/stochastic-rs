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
//! Tail dependence: $\lambda_U = 2^{-1/\theta}$ (upper),
//! $\lambda_L = 0$ (lower).
//!
//! Reference: Galambos, J. (1975), "Order statistics of samples from
//! multivariate distributions", *JASA* 70(351), 674-680.
//! Reference: Joe, H. (1997), "Multivariate Models and Dependence
//! Concepts", Chapman & Hall, §5.4.
//! Reference: Gudendorf, G. & Segers, J. (2010), "Extreme-value copulas", in Copula Theory and Its Applications, Lecture Notes in Statistics, Springer, arXiv:0911.1015, §4 (Kendall's τ from the Pickands function).

use std::sync::OnceLock;

use ndarray::Array1;
use ndarray::ArrayView2;

use crate::bivariate::BivariateFamily;
use crate::bivariate::conditional::conditional_cdf;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::invert_tau;
use crate::traits::bivariate::tau_out_of_range;

/// Clamp used to evaluate [`Galambos::partial_derivative`] just inside
/// `(0, 1)` at the `v \to 0^+` boundary instead of returning a hardcoded
/// constant there — see that method's doc for why a constant is wrong.
const BOUNDARY_EPS: f64 = 1e-12;

#[derive(Debug, Clone)]
pub struct Galambos {
  pub r#type: BivariateFamily,
  pub theta: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Default for Galambos {
  fn default() -> Self {
    Self {
      r#type: BivariateFamily::Galambos,
      theta: None,
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
}

impl BivariateExt for Galambos {
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

  /// Density via the closed-form chain-rule derivation
  /// $c(u,v) = C(u,v)/(uv) \cdot [g(x,y) + (\theta+1)\,T^{2\theta+1}/(xy)^{\theta+1}]$
  /// where $x = -\ln u$, $y = -\ln v$, $T = (x^{-\theta} + y^{-\theta})^{-1/\theta}$,
  /// $g(x,y) = ((T/x)^{\theta+1} - 1)((T/y)^{\theta+1} - 1)$.
  fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
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
  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
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

  /// $\partial_v C(u,v) = (C/v) \cdot [1 - (T/y)^{\theta+1}]$, $x, y = -\ln u, -\ln v$, $T = (x^{-\theta} + y^{-\theta})^{-1/\theta}$:
  /// 0 at `v = 1`, the formula at `v` floored to `1e-12` near 0 (a `θ`- and `u`-dependent limit), NaN outside `[0, 1]`.
  fn partial_derivative(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    Ok(conditional_cdf(x, |u, v| {
      if v >= 1.0 {
        return 0.0;
      }
      let v = v.max(BOUNDARY_EPS);
      let xx = -u.ln();
      let yy = -v.ln();
      let big_t = (xx.powf(-theta) + yy.powf(-theta)).powf(-1.0 / theta);
      let cuv = u * v * big_t.exp();
      let ratio = (big_t / yy).powf(theta + 1.0);
      (cuv / v) * (1.0 - ratio)
    }))
  }

  /// The extreme-value identity `τ = ∫₀¹ t(1 − t) A''(t)/A(t) dt` (Gudendorf & Segers 2010, §4) by 512-panel Simpson
  /// with a central-difference `A''` of step `1e-4`; 0 for `θ ≤ 1e-9`.
  fn tau_from_theta(&self, theta: f64) -> f64 {
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
      let a_pp = (Self::pickands(theta, (t + h_fd).min(1.0 - 1e-12)) - 2.0 * a
        + Self::pickands(theta, (t - h_fd).max(1e-12)))
        / (h_fd * h_fd);
      let w = if k % 2 == 0 { 2.0 } else { 4.0 };
      acc += w * t * (1.0 - t) * a_pp / a.max(1e-15);
    }
    (h_grid / 3.0) * acc
  }

  /// Brent on `ln θ` over `[1e-8, 65]`: the rule holds `τ` to `2.5e-7` up to `θ = 65` and loses it beyond (`5.7e-5` at
  /// `θ = 100`), so `0 < τ < τ(65) ≈ 0.98478` is the reachable range; `θ` to a relative `1e-8`, past the rule's resolution.
  fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError> {
    static RANGE: OnceLock<(f64, f64)> = OnceLock::new();
    let (lo, hi) = (1e-8_f64.ln(), 65.0_f64.ln());
    let (tau_lo, tau_hi) =
      *RANGE.get_or_init(|| (self.tau_from_theta(lo.exp()), self.tau_from_theta(hi.exp())));
    if !(tau > tau_lo && tau < tau_hi) {
      return Err(tau_out_of_range(
        tau,
        format!("{tau_lo:?} < tau < {tau_hi:?}"),
      ));
    }
    invert_tau(|x| self.tau_from_theta(x.exp()) - tau, lo, hi, 1e-8).map(f64::exp)
  }

  /// `λ_U = 2^{−1/θ} = 2(1 − A(1/2))` of [`Galambos::pickands`], `λ_L = 0` (Galambos 1975; Joe 1997, §5.4).
  fn tail_dependence(&self) -> TailDependence<f64> {
    self.assert_theta_valid_for_tail_dependence();
    let theta = self.theta.unwrap();
    TailDependence {
      lower: 0.0,
      upper: 2.0_f64.powf(-1.0 / theta),
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
  fn galambos_marginal_recovers_input() {
    let mut c = Galambos::new();
    c.set_theta(2.0);
    let x = array![[0.5_f64, 1.0], [1.0, 0.7]];
    let cdf = c.cdf(x.view()).unwrap();
    assert!(approx(cdf[0], 0.5, 1e-12));
    assert!(approx(cdf[1], 0.7, 1e-12));
  }

  #[test]
  fn galambos_zero_theta_limit_is_independence() {
    let mut c = Galambos::new();
    c.set_theta(1e-6);
    let x = array![[0.3_f64, 0.7], [0.5, 0.5]];
    let cdf = c.cdf(x.view()).unwrap();
    assert!(approx(cdf[0], 0.21, 1e-3), "got {}", cdf[0]);
    assert!(approx(cdf[1], 0.25, 1e-3), "got {}", cdf[1]);
  }

  #[test]
  fn galambos_large_theta_approaches_comonotone() {
    let mut c = Galambos::new();
    c.set_theta(50.0);
    let x = array![[0.3_f64, 0.7], [0.5, 0.5]];
    let cdf = c.cdf(x.view()).unwrap();
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
  fn galambos_theta_from_tau_inverts_tau() {
    let target_theta = 1.5_f64;
    let c = Galambos::new();
    let tau = c.tau_from_theta(target_theta);
    let recovered = c.theta_from_tau(tau).unwrap();
    assert!(
      approx(recovered, target_theta, 1e-8),
      "expected {target_theta}, got {recovered}"
    );
  }

  #[test]
  fn galambos_pdf_positive_interior() {
    let mut c = Galambos::new();
    c.set_theta(1.5);
    let x = array![[0.25_f64, 0.75], [0.5, 0.5], [0.1, 0.9]];
    let pdf = c.pdf(x.view()).unwrap();
    for &p in pdf.iter() {
      assert!(p > 0.0 && p.is_finite(), "pdf={p}");
    }
  }

  #[test]
  fn galambos_tail_dependence_matches_pickands_formula() {
    let mut c = Galambos::new();
    c.set_theta(1.5);
    let td = c.tail_dependence();
    let expected = 2.0 * (1.0 - Galambos::pickands(1.5, 0.5));
    assert!(approx(td.upper, expected, 1e-12), "got {}", td.upper);
    assert_eq!(td.lower, 0.0);
  }

  /// `partial_derivative` must not jump to a hardcoded constant right at
  /// the `v\to0^+` boundary: its value at `v=1e-12` should be continuous
  /// with its value at `v=1e-9` (the F8 regression this guards). Checked
  /// at `\theta=1.5`, where the pre-fix code hardcoded `0.0` even though
  /// the true limit there is close to `1` for this `\theta` (see the
  /// method's doc).
  ///
  /// Tolerance `1e-2`, not `1e-6`: measured directly against this formula
  /// at `\theta=1.5,u=0.4`, the real gap between `v=1e-12` and `v=1e-9` is
  /// `\approx 2.2e-3` (this family's boundary convergence in `v`, like
  /// Hüsler-Reiss's, is not fast enough for `1e-6` to hold across a
  /// `1000\times` shrink in `v`). `1e-2` still rejects a jump to a
  /// hardcoded constant by more than two orders of magnitude (such a jump
  /// is `\approx 0.6` here, since the true limit sits near `1`, not `0`).
  #[test]
  fn galambos_partial_derivative_no_jump_near_v_zero() {
    let mut c = Galambos::new();
    c.set_theta(1.5);
    let u = 0.4_f64;
    let a = c.partial_derivative(array![[u, 1e-12]].view()).unwrap()[0];
    let b = c.partial_derivative(array![[u, 1e-9]].view()).unwrap()[0];
    assert!(
      approx(a, b, 1e-2),
      "expected continuity near v=0: v=1e-12 -> {a}, v=1e-9 -> {b}"
    );
    assert!(
      a > 0.9,
      "theta=1.5 v->0+ limit should be close to 1, got {a}"
    );
  }

  /// `generator` has no override in this family, so this exercises
  /// `BivariateExt::generator`'s trait-default body directly.
  #[test]
  fn galambos_generator_returns_err_not_archimedean() {
    let c = Galambos::new();
    let t = array![0.5_f64, 0.8];
    assert!(c.generator(t.view()).is_err());
  }
}
