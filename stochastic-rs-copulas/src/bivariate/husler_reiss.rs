//! # Hüsler-Reiss (1989) extreme-value copula
//!
//! ```text
//! C_λ(u,v) = exp{ ln u · Φ(1/λ + (λ/2)·ln(ln u / ln v))
//!                + ln v · Φ(1/λ + (λ/2)·ln(ln v / ln u)) },   λ ∈ [0, ∞),
//! A(t) = (1-t)·Φ(1/λ + (λ/2)·ln((1-t)/t)) + t·Φ(1/λ + (λ/2)·ln(t/(1-t))),   t ∈ [0, 1].
//! ```
//!
//! `λ = 0` is independence (`A ≡ 1`), `λ → ∞` comonotonicity; `λ_U = 2(1 − Φ(1/λ))`, `λ_L = 0`. The identity
//! `x φ(α) = y φ(β)`, `α = 1/λ + (λ/2) ln(x/y)`, `β = 2/λ − α`, collapses `∂_y ln C` to `−Φ(β)`.
//!
//! Reference: Hüsler, J., Reiss, R.-D. (1989), "Maxima of normal random
//! vectors: between independence and complete dependence", *Statist.
//! Probab. Lett.* 7(4), 283-286.
//! Reference: Gudendorf, G. & Segers, J. (2010), "Extreme-value copulas", in Copula Theory and Its Applications, Lecture Notes in Statistics, Springer, arXiv:0911.1015, §3.3 (the parameter space, `[0, ∞]` in the reciprocal of this `λ`, independence at its `∞`), §4 (Kendall's τ from the Pickands function).

use std::f64::consts::LN_2;
use std::f64::consts::TAU;

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_distributions::special::norm_pdf;

use crate::bivariate::BivariateFamily;
use crate::bivariate::conditional::conditional_cdf;
use crate::bivariate::conditional::conditional_quantiles;
use crate::bivariate::conditional::copula_cdf;
use crate::bivariate::conditional::copula_pdf;
use crate::bivariate::conditional::ln_add_exp;
use crate::bivariate::conditional::ln_log_ratio;
use crate::bivariate::conditional::ln_norm_cdf;
use crate::bivariate::conditional::ln_one_minus_exp;
use crate::bivariate::conditional::mills;
use crate::bivariate::conditional::numerical_quantiles;
use crate::bivariate::conditional::on_open_square;
use crate::bivariate::conditional::softplus;
use crate::bivariate::integral::ln_integral_from_peak;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::admitted;
use crate::traits::bivariate::checked_percent_point;
use crate::traits::bivariate::invert_tau;
use crate::traits::bivariate::tau_out_of_range;

/// The ends of the τ inversion's bracket in `λ`: below `1e-2`, `τ < e^{−4900}`, and past `1e17`, `1 − τ < 2⁻⁵⁴`.
const LAMBDA_BRACKET: (f64, f64) = (1e-2, 1e17);

#[derive(Debug, Clone, Default)]
pub struct HuslerReiss {
  theta: Option<f64>,
}

impl HuslerReiss {
  /// A Hüsler–Reiss copula with `lambda`, stored as `theta`; panics unless `0 <= lambda < ∞`.
  pub fn new(lambda: f64) -> Self {
    Self::try_new(lambda).unwrap_or_else(|e| panic!("{e}"))
  }

  /// `new` with the domain error returned instead of a panic.
  pub fn try_new(lambda: f64) -> Result<Self, CopulaError> {
    let copula = Self {
      theta: Some(lambda),
    };
    copula.check_theta()?;
    Ok(copula)
  }

  /// The copula whose Kendall's τ is `tau`.
  pub fn from_tau(tau: f64) -> Result<Self, CopulaError> {
    Self::try_new(Self::default().theta_from_tau(tau)?)
  }

  /// `ln ∂_v C = −xΦ(α) + yΦ(−β) + ln Φ(β)`, `x, y = −ln u, −ln v`, `α, β = 1/λ ± (λ/2) ln(x/y)`, `ln(x/y)` by `ln_log_ratio`,
  /// and every normal tail through Mills' ratio, so `1 − h` keeps its digits; the limit laws `δ₀`, `δ₁` at `v = 0, 1`.
  pub(crate) fn ln_h(lambda: f64) -> impl Fn(f64, f64) -> f64 {
    let inv = 1.0 / lambda;
    move |u, v| {
      if u >= 1.0 || v <= 0.0 {
        return 0.0;
      }
      if u <= 0.0 || v >= 1.0 {
        return f64::NEG_INFINITY;
      }
      let (x, y) = (-u.ln(), -v.ln());
      let half = 0.5 * lambda * ln_log_ratio(u, v, x, y);
      let (alpha, beta) = (inv + half, inv - half);
      // For `α < 0 < β` both terms are tails that nearly cancel; `xφ(α) = yφ(β)` turns them into one Mills difference.
      let tails = if alpha < 0.0 {
        y * norm_pdf(beta) * (mills(beta) - mills(-alpha))
      } else {
        y * norm_cdf(-beta) - x * norm_cdf(alpha)
      };
      tails + ln_norm_cdf(beta)
    }
  }

  /// `(x, y, α, β)`, `x, y = −ln u, −ln v`, `α, β = 1/λ ± (λ/2) ln(x/y)` with `ln(x/y)` by `ln_log_ratio`.
  fn scores(lambda: f64, u: f64, v: f64) -> (f64, f64, f64, f64) {
    let (x, y) = (-u.ln(), -v.ln());
    let half = 0.5 * lambda * ln_log_ratio(u, v, x, y);
    (x, y, 1.0 / lambda + half, 1.0 / lambda - half)
  }

  /// `ln c = xΦ(−α) + yΦ(−β) + ln(Φ(α)Φ(β) + (λ/2)φ(α)/y)`, the last a log-sum of positive terms, each normal tail by
  /// `ln_norm_cdf`, so the density holds where both terms underflow.
  fn ln_density(lambda: f64, u: f64, v: f64) -> f64 {
    if lambda == 0.0 {
      return 0.0;
    }
    let (x, y, alpha, beta) = Self::scores(lambda, u, v);
    let ln_phi = -0.5 * alpha * alpha - 0.5 * std::f64::consts::TAU.ln();
    x * norm_cdf(-alpha)
      + y * norm_cdf(-beta)
      + ln_add_exp(
        ln_norm_cdf(alpha) + ln_norm_cdf(beta),
        lambda.ln() - std::f64::consts::LN_2 + ln_phi - y.ln(),
      )
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

  /// `(ln τ, ln(1 − τ))`: as `A' = Φ(β) − Φ(α)`, `τ = λ∫ (1 − t)φ(α)/A` and, by parts, `1 − τ = 2∫ t(1 − t)Φ(α)Φ(β)/A²`
  /// over the logit `ℓ` of `t ∈ [1/2, 1]`, `α, β = 1/λ ∓ λℓ/2`; the smaller is integrated, the other follows.
  fn ln_tau_parts(lambda: f64) -> (f64, f64) {
    // `(ln(1 − t), ln t, α, β, ln A)` at `ℓ`.
    let at = |l: f64| {
      let (alpha, beta) = (
        1.0 / lambda - 0.5 * lambda * l,
        1.0 / lambda + 0.5 * lambda * l,
      );
      let (ln_one_minus_t, ln_t) = (-softplus(l), -softplus(-l));
      let a = ln_one_minus_t.exp() * norm_cdf(alpha) + ln_t.exp() * norm_cdf(beta);
      (ln_one_minus_t, ln_t, alpha, beta, a.ln())
    };
    if lambda <= 2.0 {
      let ln_tau = ln_integral_from_peak(
        |l: f64| {
          let (ln_one_minus_t, _, alpha, _, ln_a) = at(l);
          lambda.ln() + ln_one_minus_t - 0.5 * (alpha * alpha + TAU.ln()) - ln_a
        },
        0.5,
      );
      (ln_tau, ln_one_minus_exp(ln_tau))
    } else {
      // In `w = λℓ`, the scale of `Φ(α)`'s fall.
      let ln_complement = LN_2 - lambda.ln()
        + ln_integral_from_peak(
          |w: f64| {
            let (ln_one_minus_t, ln_t, alpha, beta, ln_a) = at(w / lambda);
            ln_one_minus_t + ln_t + ln_norm_cdf(alpha) + ln_norm_cdf(beta) - 2.0 * ln_a
          },
          0.5,
        );
      (ln_one_minus_exp(ln_complement), ln_complement)
    }
  }

  /// `numerical_quantiles` of `HuslerReiss::ln_h`, `y` at `λ = 0`.
  fn unchecked_inverse(
    &self,
    y: ArrayView1<f64>,
    v: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    let lambda = self.theta.unwrap();
    if lambda == 0.0 {
      return Ok(conditional_quantiles(y, v, |y, _| y));
    }
    numerical_quantiles(y, v, self.family(), Self::ln_h(lambda))
  }
}

impl BivariateExt for HuslerReiss {
  fn family(&self) -> BivariateFamily {
    BivariateFamily::HuslerReiss
  }

  fn theta(&self) -> Option<f64> {
    self.theta
  }

  fn set_theta(&mut self, theta: f64) {
    self.validate_theta(theta).unwrap_or_else(|e| panic!("{e}"));
    self.theta = Some(theta);
  }

  fn theta_bounds(&self) -> (f64, f64) {
    (0.0, f64::INFINITY)
  }

  fn invalid_thetas(&self) -> Vec<f64> {
    vec![]
  }

  fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let lambda = self.theta.unwrap();
    copula_pdf(x, |u, v| Self::ln_density(lambda, u, v))
  }

  fn log_pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let lambda = self.theta.unwrap();
    on_open_square(x, |u, v| Self::ln_density(lambda, u, v))
  }

  /// `C = e^{−xΦ(α) − yΦ(β)}` from `HuslerReiss::scores`; `uv` at `λ = 0`.
  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let lambda = self.theta.unwrap();
    if lambda == 0.0 {
      return copula_cdf(x, |u, v| u * v);
    }
    copula_cdf(x, |u, v| {
      let (x, y, alpha, beta) = Self::scores(lambda, u, v);
      (-x * norm_cdf(alpha) - y * norm_cdf(beta)).exp()
    })
  }

  /// `exp` of `HuslerReiss::ln_h`, `u` at `λ = 0`.
  fn partial_derivative(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let lambda = self.theta.unwrap();
    if lambda == 0.0 {
      return conditional_cdf(x, |u, _| u);
    }
    let ln_h = Self::ln_h(lambda);
    conditional_cdf(x, |u, v| ln_h(u, v).exp())
  }

  fn percent_point(
    &self,
    y: ArrayView1<f64>,
    v: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    checked_percent_point(self, y, v, Self::unchecked_inverse)
  }

  fn percent_point_unchecked(
    &self,
    y: ArrayView1<f64>,
    v: ArrayView1<f64>,
  ) -> Option<Result<Array1<f64>, CopulaError>> {
    Some(self.unchecked_inverse(y, v))
  }

  /// Gudendorf & Segers' (2010, §4) `τ = ∫₀¹ t(1 − t)/A dA'` through `HuslerReiss::ln_tau_parts`; 0 below `λ = 1e-2`.
  fn tau_from_theta(&self, lambda: f64) -> f64 {
    if lambda < LAMBDA_BRACKET.0 {
      return 0.0;
    }
    let (ln_tau, ln_complement) = Self::ln_tau_parts(lambda);
    if ln_tau < ln_complement {
      ln_tau.exp()
    } else {
      -ln_complement.exp_m1()
    }
  }

  /// Brent on `ln λ` over `LAMBDA_BRACKET`, matching `ln τ` up to `τ = 1/2` and `ln(1 − τ)` past it, so every `τ` in
  /// `[0, 1)`, subnormal or a few ulps under 1, inverts to its relative digits.
  fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError> {
    if !(0.0..1.0).contains(&tau) {
      return Err(tau_out_of_range(tau, "0.0 <= tau < 1.0".to_string()));
    }
    let theta = if tau == 0.0 {
      0.0
    } else {
      let residual = |x: f64| {
        let (ln_tau, ln_complement) = Self::ln_tau_parts(x.exp());
        if tau <= 0.5 {
          ln_tau - tau.ln()
        } else {
          (-tau).ln_1p() - ln_complement
        }
      };
      let (lo, hi) = LAMBDA_BRACKET;
      invert_tau(residual, lo.ln(), hi.ln(), 1e-13)?.exp()
    };
    admitted(self, tau, theta)
  }

  /// Upper-tail dependence $\lambda_U = 2(1 - \Phi(1/\lambda))$;
  /// Hüsler-Reiss has no lower-tail dependence.
  /// Reference: Hüsler, J., Reiss, R.-D. (1989) (module header).
  fn tail_dependence(&self) -> Result<TailDependence<f64>, CopulaError> {
    self.check_theta()?;
    let lambda = self.theta.unwrap();
    Ok(TailDependence {
      lower: 0.0,
      upper: 2.0 * (1.0 - norm_cdf(1.0 / lambda)),
    })
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// `(ln τ, ln(1 − τ))`, the logarithm each inversion matches, against the τ table's second and third columns: each
  /// within `1e-12` of itself, or `f64::MIN_POSITIVE` where it is subnormal.
  #[test]
  fn the_tau_logarithms_match_the_tau_table() {
    let rows = include_str!("../../tests/fixtures/kendall_tau.txt")
      .lines()
      .filter(|l| l.starts_with("husler_reiss "));
    for line in rows {
      let (head, values) = line.split_once(" ; ").unwrap();
      let theta: f64 = head[13..].parse().unwrap();
      let want: Vec<f64> = values
        .split(' ')
        .skip(1)
        .map(|x| x.parse().unwrap())
        .collect();
      if want[0] == f64::NEG_INFINITY {
        continue;
      }
      let (ln_tau, ln_complement) = HuslerReiss::ln_tau_parts(theta);
      for (got, want) in [(ln_tau, want[0]), (ln_complement, want[1])] {
        let tolerance = (1e-12 * want.abs()).max(f64::MIN_POSITIVE);
        assert!(
          (got - want).abs() <= tolerance,
          "θ = {theta:e}: {got:e}, want {want:e}"
        );
      }
    }
  }

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  #[test]
  fn hr_marginal_recovers_input() {
    let mut c = HuslerReiss::default();
    c.set_theta(1.5);
    let x = array![[0.5_f64, 1.0], [1.0, 0.7]];
    let cdf = c.cdf(x.view()).unwrap();
    assert!(approx(cdf[0], 0.5, 1e-12));
    assert!(approx(cdf[1], 0.7, 1e-12));
  }

  #[test]
  fn hr_small_lambda_limit_is_independence() {
    let mut c = HuslerReiss::default();
    c.set_theta(1e-3);
    let x = array![[0.3_f64, 0.7], [0.5, 0.5]];
    let cdf = c.cdf(x.view()).unwrap();
    assert!(approx(cdf[0], 0.21, 1e-3), "got {}", cdf[0]);
    assert!(approx(cdf[1], 0.25, 1e-3), "got {}", cdf[1]);
  }

  #[test]
  fn hr_large_lambda_approaches_comonotone() {
    let mut c = HuslerReiss::default();
    c.set_theta(15.0);
    let x = array![[0.3_f64, 0.7], [0.5, 0.5]];
    let cdf = c.cdf(x.view()).unwrap();
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
  fn hr_theta_from_tau_inverts_tau() {
    let target_lambda = 1.5_f64;
    let c = HuslerReiss::default();
    let tau = c.tau_from_theta(target_lambda);
    let recovered = c.theta_from_tau(tau).unwrap();
    assert!(
      approx(recovered, target_lambda, 1e-12),
      "expected {target_lambda}, got {recovered}"
    );
  }

  #[test]
  fn hr_pdf_positive_interior() {
    let mut c = HuslerReiss::default();
    c.set_theta(1.5);
    let x = array![[0.25_f64, 0.75], [0.5, 0.5], [0.1, 0.9]];
    let pdf = c.pdf(x.view()).unwrap();
    for &p in pdf.iter() {
      assert!(p > 0.0 && p.is_finite(), "pdf={p}");
    }
  }

  #[test]
  fn hr_tail_dependence_matches_formula() {
    let mut c = HuslerReiss::default();
    c.set_theta(1.5);
    let td = c.tail_dependence().unwrap();
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
    let mut c = HuslerReiss::default();
    c.set_theta(1.5);
    let u = 0.4_f64;
    let near_zero_a = c.partial_derivative(array![[u, 1e-12]].view()).unwrap()[0];
    let near_zero_b = c.partial_derivative(array![[u, 1e-9]].view()).unwrap()[0];
    assert!(
      approx(near_zero_a, near_zero_b, 2e-2),
      "expected continuity near v=0: v=1e-12 -> {near_zero_a}, v=1e-9 -> {near_zero_b}"
    );
    let near_one_a = c
      .partial_derivative(array![[u, 1.0 - 1e-12]].view())
      .unwrap()[0];
    let near_one_b = c
      .partial_derivative(array![[u, 1.0 - 1e-9]].view())
      .unwrap()[0];
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

  /// `λ = 0` is independence: `C = uv`, `c = 1` and `∂_v C = u` on the closed square, where the general formula's `0·∞`
  /// is NaN at `u ∈ {0, 1}` and the numerical inverse, so sampling, failed.
  #[test]
  fn hr_lambda_zero_is_independence() {
    let c = HuslerReiss::new(0.0);
    let x = array![[0.3_f64, 0.7], [0.2, 0.9]];
    assert_eq!(c.cdf(x.view()).unwrap(), array![0.3 * 0.7, 0.2 * 0.9]);
    assert_eq!(c.pdf(x.view()).unwrap(), array![1.0, 1.0]);
    let edges = array![[0.0, 0.5], [1.0, 0.5], [0.3, 0.5], [0.3, 0.0], [0.3, 1.0]];
    let h = c.partial_derivative(edges.view()).unwrap();
    assert_eq!(h, array![0.0, 1.0, 0.3, 0.3, 0.3]);
    assert_eq!(c.tail_dependence().unwrap().upper, 0.0);
    let n = 4000_usize;
    let uv = c.sample(n, &Deterministic::new(42)).unwrap();
    let (tau, ..) = kendalls::tau_b_with_comparator(
      &uv.column(0).to_vec(),
      &uv.column(1).to_vec(),
      |a: &f64, b: &f64| a.partial_cmp(b).unwrap(),
    )
    .unwrap();
    // Six standard errors of τ under independence (Kendall & Gibbons 1990).
    let se = (2.0 * (2.0 * n as f64 + 5.0) / (9.0 * n as f64 * (n as f64 - 1.0))).sqrt();
    assert!(tau.abs() < 6.0 * se, "τ = {tau}");
  }

  /// `τ = 0` inverts to `λ = 0`, the independence copula `new(0.0)` builds, from `from_tau` and from a fit at `τ̂ = 0`.
  #[test]
  fn hr_tau_zero_inverts_to_independence() {
    let c = HuslerReiss::from_tau(0.0).unwrap();
    assert_eq!((c.theta(), c.tau()), (Some(0.0), Some(0.0)));
    let ranks = [2.0, 4.0, 1.0, 3.0];
    let tied = ndarray::Array2::from_shape_fn((4, 2), |(i, j)| {
      if j == 0 {
        (i + 1) as f64 / 5.0
      } else {
        ranks[i] / 5.0
      }
    });
    let mut fitted = HuslerReiss::default();
    fitted.fit(tied.view()).unwrap();
    assert_eq!(fitted.theta(), Some(0.0));
    assert!(matches!(
      HuslerReiss::from_tau(-1e-12),
      Err(CopulaError::InvalidParameter { name: "tau", .. })
    ));
  }

  /// `generator` has no override in this family, so this exercises
  /// `BivariateExt::generator`'s trait-default body directly.
  #[test]
  fn hr_generator_returns_err_not_archimedean() {
    let c = HuslerReiss::default();
    let t = array![0.5_f64, 0.8];
    assert!(c.generator(t.view()).is_err());
  }
}
