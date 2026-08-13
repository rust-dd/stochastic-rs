//! # Bivariate Gaussian copula
//!
//! $$
//! C_\rho(u,v) = \Phi_2\big(\Phi^{-1}(u), \Phi^{-1}(v); \rho\big), \qquad \rho \in (-1, 1),
//! $$
//! where $\Phi_2(\cdot,\cdot;\rho)$ is the standard bivariate normal CDF with
//! correlation $\rho$ and $\Phi^{-1}$ the standard normal quantile.
//!
//! - **Kendall's tau:** $\tau = \tfrac{2}{\pi}\arcsin\rho$, inverted in
//!   closed form as $\rho = \sin(\pi\tau/2)$.
//!   Reference: Kruskal, W.H. (1958), "Ordinal Measures of Association",
//!   *JASA* 53(284), 814-861.
//! - **Tail dependence:** $\lambda_L = \lambda_U = 0$ for every $\rho \in
//!   (-1, 1)$ — the Gaussian copula is asymptotically independent in both
//!   tails regardless of $\rho$, in contrast to the Student-t copula
//!   ([`crate::bivariate::t_copula::TCopula`]), which has strictly positive
//!   tail dependence for every finite degrees-of-freedom parameter.
//!
//! The joint CDF is evaluated through the identity
//! $$
//! \Phi_2(x,y;\rho) = \Phi(x) + \Phi(y) - 1 + Q(x,y;\rho),
//! $$
//! where $Q(x,y;\rho) = \Pr\left[X>x, Y>y\right]$ is the bivariate normal upper
//! survival probability computed by `owens_t::biv_norm` (an Owen's
//! T-function reduction) — the crate exposes the survival form, not the
//! CDF directly, so the inclusion-exclusion identity above bridges the two.
//!
//! Reference: Embrechts, P., Lindskog, F., McNeil, A.J. (2003),
//! "Modelling Dependence with Copulas and Applications to Risk
//! Management", in *Handbook of Heavy Tailed Distributions in Finance*,
//! Elsevier, ch. 8.
//! Reference: Owen, D.B. (1956), "Tables for computing bivariate normal
//! probabilities", *Ann. Math. Statist.* 27(4), 1075-1090.

use std::error::Error;
use std::f64;

use ndarray::Array1;
use ndarray::Array2;
use owens_t::biv_norm;
use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::special::norm_cdf;

use crate::bivariate::CopulaType;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;

/// Clamp used to evaluate [`GaussianCopula::partial_derivative`] just
/// inside `(0, 1)` at the `v` boundary instead of returning a hardcoded
/// constant there — see that method's doc for why a constant is wrong.
const BOUNDARY_EPS: f64 = 1e-12;

#[derive(Debug, Clone)]
pub struct GaussianCopula {
  pub r#type: CopulaType,
  /// Correlation $\rho \in (-1, 1)$, stored under the trait's `theta`
  /// field for single-parameter compatibility with the other families.
  pub theta: Option<f64>,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Default for GaussianCopula {
  fn default() -> Self {
    Self {
      r#type: CopulaType::Gaussian,
      theta: None,
      tau: None,
      theta_bounds: (-1.0, 1.0),
      invalid_thetas: vec![-1.0, 1.0],
    }
  }
}

impl GaussianCopula {
  pub fn new() -> Self {
    Self::default()
  }
}

impl BivariateExt for GaussianCopula {
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

  /// Density $c(u,v) = (1-\rho^2)^{-1/2}\exp\{-(\rho^2(x^2+y^2) -
  /// 2\rho xy)/(2(1-\rho^2))\}$ with $x = \Phi^{-1}(u)$, $y = \Phi^{-1}(v)$
  /// — the ratio of the bivariate normal density to the product of its
  /// marginals.
  fn pdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;
    let rho = self.theta.unwrap();
    let one_minus_rho2 = 1.0 - rho * rho;
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
      let xx = ndtri(u);
      let yy = ndtri(v);
      let quad = rho * rho * (xx * xx + yy * yy) - 2.0 * rho * xx * yy;
      out[i] = (-quad / (2.0 * one_minus_rho2)).exp() / one_minus_rho2.sqrt();
    }
    Ok(out)
  }

  /// $C(u,v) = \Phi_2(x,y;\rho) = \Phi(x) + \Phi(y) - 1 + Q(x,y;\rho)$,
  /// with $Q$ the `owens_t::biv_norm` survival probability (see module
  /// header for the inclusion-exclusion derivation).
  fn cdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;
    let rho = self.theta.unwrap();
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
      let xx = ndtri(u);
      let yy = ndtri(v);
      out[i] = norm_cdf(xx) + norm_cdf(yy) - 1.0 + biv_norm(xx, yy, rho);
    }
    Ok(out)
  }

  /// $\partial_v C(u,v) = \Phi\big((x - \rho y)/\sqrt{1-\rho^2}\big)$, the
  /// bivariate-normal conditional CDF $\Pr[X\le x \mid Y=y]$ with $x =
  /// \Phi^{-1}(u)$, $y = \Phi^{-1}(v)$. Same "derivative w.r.t. the second
  /// argument, at fixed conditioning value" convention as
  /// [`crate::bivariate::clayton::Clayton::partial_derivative`].
  ///
  /// At the `v \to 0^+/1^-` boundary, `y = \Phi^{-1}(v) \to \mp\infty`, so
  /// `(x-\rho y)/\sqrt{1-\rho^2} \to \pm\infty` when `\rho>0` but `\to
  /// \mp\infty` when `\rho<0` — the boundary value's sign flips with the
  /// sign of `\rho` — and at `\rho=0` the `\rho y` term vanishes
  /// identically, leaving `\Phi(x)=u`, which depends on `u` rather than
  /// being a constant. A single hardcoded `0.0`/`1.0` is therefore wrong
  /// for at least one sign of `\rho` (and always wrong at `\rho=0`);
  /// evaluating the same formula at `v` clamped just inside `(0,1)` instead
  /// gives the mathematically correct directional limit (by continuity of
  /// `\Phi`/`\Phi^{-1}` on the open interval) without a family of
  /// hardcoded special cases.
  fn partial_derivative(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;
    let rho = self.theta.unwrap();
    let sqrt_one_minus_rho2 = (1.0 - rho * rho).sqrt();
    let u_col = x.column(0);
    let v_col = x.column(1);
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i].clamp(BOUNDARY_EPS, 1.0 - BOUNDARY_EPS);
      let xx = ndtri(u);
      let yy = ndtri(v);
      out[i] = norm_cdf((xx - rho * yy) / sqrt_one_minus_rho2);
    }
    Ok(out)
  }

  /// Closed-form inverse of [`GaussianCopula::partial_derivative`] w.r.t.
  /// $u$ at fixed $v$: solving $\Phi((\Phi^{-1}(u) - \rho y)/\sqrt{1-\rho^2})
  /// = p$ for $u$ gives $u = \Phi(\Phi^{-1}(p)\sqrt{1-\rho^2} + \rho y)$,
  /// with $y = \Phi^{-1}(v)$.
  fn percent_point(&self, y: &Array1<f64>, V: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;
    let rho = self.theta.unwrap();
    let sqrt_one_minus_rho2 = (1.0 - rho * rho).sqrt();
    let n = y.len();
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
      let p = y[i];
      let yy = ndtri(V[i]);
      out[i] = norm_cdf(ndtri(p) * sqrt_one_minus_rho2 + rho * yy);
    }
    Ok(out)
  }

  /// Closed-form Kendall's tau inversion $\rho = \sin(\pi\tau/2)$.
  /// Reference: Kruskal, W.H. (1958), *JASA* 53(284), 814-861.
  fn compute_theta(&self) -> f64 {
    let tau = self.tau.unwrap();
    (0.5 * std::f64::consts::PI * tau).sin().clamp(-1.0, 1.0)
  }

  /// Asymptotic independence: $\lambda_L = \lambda_U = 0$ for every $\rho
  /// \in (-1, 1)$ — the classical counterexample to "correlation implies
  /// tail dependence".
  /// Reference: Embrechts, Lindskog, McNeil (2003), §3.2 (module header).
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
  use ndarray::array;

  use super::*;

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  #[test]
  fn gaussian_marginal_recovers_input() {
    let mut c = GaussianCopula::new();
    c.set_theta(0.4);
    let x = array![[0.5_f64, 1.0], [1.0, 0.7]];
    let cdf = c.cdf(&x).unwrap();
    assert!(approx(cdf[0], 0.5, 1e-9));
    assert!(approx(cdf[1], 0.7, 1e-9));
  }

  /// Φ₂(0,0;ρ) = 1/4 + asin(ρ)/(2π); ρ=0.5 → 1/3 exactly.
  ///
  /// Tolerance is `1e-6`, not the formula's own `f64` precision: `cdf`
  /// routes through [`stochastic_rs_distributions::special::norm_cdf`],
  /// whose `erf` is the Abramowitz-Stegun 7.1.26 rational approximation
  /// (~1.5e-7 relative error by its own doc comment). At this exactly
  /// symmetric point the two `norm_cdf` calls should cancel to `0.0` in
  /// `Φ(x) + Φ(y) - 1`, so the approximation's residual at `x=0` (~1e-9)
  /// leaks directly into the result.
  #[test]
  fn gaussian_cdf_matches_closed_form_at_origin() {
    let mut c = GaussianCopula::new();
    c.set_theta(0.5);
    let x = array![[0.5_f64, 0.5]];
    let cdf = c.cdf(&x).unwrap()[0];
    let expected = 0.25 + (0.5_f64).asin() / (2.0 * std::f64::consts::PI);
    assert!(
      approx(cdf, expected, 1e-6),
      "got {cdf}, expected {expected}"
    );
    assert!(approx(cdf, 1.0 / 3.0, 1e-6), "got {cdf}, expected 1/3");
  }

  /// ρ=0.6 → τ → compute_theta → ρ.
  #[test]
  fn gaussian_tau_roundtrip() {
    let rho = 0.6_f64;
    let tau = (2.0 / std::f64::consts::PI) * rho.asin();
    let mut c = GaussianCopula::new();
    c.set_tau(tau);
    let recovered = c.compute_theta();
    assert!(
      approx(recovered, rho, 1e-12),
      "got {recovered}, expected {rho}"
    );
  }

  #[test]
  fn gaussian_fit_recovers_rho_from_samples() {
    let rho_true = 0.6_f64;
    let n = 20_000usize;
    let tol = 0.02_f64;
    let mut best_err = f64::INFINITY;
    for &seed in &[2718_u64, 999, 42] {
      let mut source = GaussianCopula::new();
      source.set_theta(rho_true);
      // `sample_with_uniform`'s trait-level gate requires `tau` to be set
      // (it does not derive it from `theta`); use the value consistent
      // with `rho_true` via the forward Kendall relation.
      source.set_tau((2.0 / std::f64::consts::PI) * rho_true.asin());
      let samples = source.sample_with_seed(n, seed).unwrap();

      let mut fitted = GaussianCopula::new();
      fitted.fit(&samples).unwrap();
      let rho_hat = fitted.theta().unwrap();
      best_err = best_err.min((rho_hat - rho_true).abs());
    }
    assert!(
      best_err < tol,
      "best-of-3 |ρ̂ - ρ| = {best_err}, expected < {tol}"
    );
  }

  /// percent_point(partial_derivative(u,v), v) == u.
  ///
  /// Tolerance `1e-6` (not raw `f64` precision): the round trip crosses
  /// `norm_cdf`'s ~1.5e-7-relative-error `erf` approximation twice (once
  /// in `partial_derivative`, once in `percent_point`), see
  /// [`gaussian_cdf_matches_closed_form_at_origin`] for the same root
  /// cause.
  #[test]
  fn gaussian_h_inverse_roundtrip() {
    let mut c = GaussianCopula::new();
    c.set_theta(0.35);
    let u = 0.3_f64;
    let v = 0.65_f64;
    let p = c.partial_derivative(&array![[u, v]]).unwrap()[0];
    let u_back = c
      .percent_point(&Array1::from_vec(vec![p]), &Array1::from_vec(vec![v]))
      .unwrap()[0];
    assert!(approx(u_back, u, 1e-6), "got {u_back}, expected {u}");
  }

  #[test]
  fn gaussian_tail_dependence_is_zero() {
    let mut c = GaussianCopula::new();
    c.set_theta(0.9);
    let td = c.tail_dependence();
    assert_eq!(td.lower, 0.0);
    assert_eq!(td.upper, 0.0);
  }

  #[test]
  fn gaussian_pdf_positive_on_unit_square_interior() {
    let mut c = GaussianCopula::new();
    c.set_theta(0.3);
    let x = array![[0.25_f64, 0.75], [0.5, 0.5], [0.1, 0.9]];
    let pdf = c.pdf(&x).unwrap();
    for &p in pdf.iter() {
      assert!(p > 0.0 && p.is_finite(), "pdf={p}");
    }
  }

  #[test]
  fn gaussian_generator_returns_err_not_archimedean() {
    let c = GaussianCopula::new();
    let t = array![0.5_f64, 0.8];
    assert!(c.generator(&t).is_err());
  }

  /// `partial_derivative` must not jump to a hardcoded constant right at
  /// the `v` boundary: its value at `v=1e-12` should be continuous with
  /// its value at `v=1e-9`, tracking the true directional limit rather
  /// than snapping to a wrong-signed `0.0`/`1.0` (the F8 regression this
  /// guards). Checked at `\rho>0`, where the pre-fix code returned the
  /// limits backwards (`0.0` near `v=0`, `1.0` near `v=1`, when the true
  /// direction is the opposite — see the method's doc).
  #[test]
  fn gaussian_partial_derivative_no_jump_near_v_boundary() {
    let mut c = GaussianCopula::new();
    c.set_theta(0.7);
    let u = 0.4_f64;
    let near_zero_a = c.partial_derivative(&array![[u, 1e-12]]).unwrap()[0];
    let near_zero_b = c.partial_derivative(&array![[u, 1e-9]]).unwrap()[0];
    assert!(
      approx(near_zero_a, near_zero_b, 1e-6),
      "expected continuity near v=0: v=1e-12 -> {near_zero_a}, v=1e-9 -> {near_zero_b}"
    );
    let near_one_a = c.partial_derivative(&array![[u, 1.0 - 1e-12]]).unwrap()[0];
    let near_one_b = c.partial_derivative(&array![[u, 1.0 - 1e-9]]).unwrap()[0];
    assert!(
      approx(near_one_a, near_one_b, 1e-6),
      "expected continuity near v=1: v=1-1e-12 -> {near_one_a}, v=1-1e-9 -> {near_one_b}"
    );

    // Directional regression check: for rho > 0, the true limit is *high*
    // near v=0 and *low* near v=1 — the opposite of the pre-fix hardcoded
    // 0.0/1.0.
    assert!(
      near_zero_a > 0.99,
      "rho>0 should push the v->0+ limit toward 1, got {near_zero_a}"
    );
    assert!(
      near_one_a < 0.01,
      "rho>0 should push the v->1- limit toward 0, got {near_one_a}"
    );
  }
}
