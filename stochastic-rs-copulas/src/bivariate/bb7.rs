//! # BB7
//!
//! $$
//! C_{\theta,\delta}(u,v)=1-\Bigl(1-\bigl[(1-\bar u^\theta)^{-\delta}+(1-\bar v^\theta)^{-\delta}-1\bigr]^{-1/\delta}\Bigr)^{1/\theta},\qquad \bar u=1-u,\ \theta\ge1,\ \delta>0
//! $$
//!
//! The Joe–Clayton copula: Archimedean with generator
//! `φ(t) = (1 − (1 − t)^θ)^{−δ} − 1`, upper tail from the Joe part and lower
//! tail from the Clayton part, `λ_U = 2 − 2^{1/θ}`, `λ_L = 2^{−1/δ}`. With
//! `a = 1 − ū^θ`, `b = 1 − v̄^θ`, `s = a^{−δ} + b^{−δ} − 1` and `T = s^{−1/δ}`:
//! `∂_v C = (1 − T)^{1/θ−1} s^{−1/δ−1} b^{−δ−1} v̄^{θ−1}` and
//! `c = (1 − T)^{1/θ−2} s^{−1/δ−2} (ab)^{−δ−1} (ū v̄)^{θ−1} [θ(δ + 1) − (θδ + 1) T]`.
//!
//! Kendall's τ is a Beta form from Nelsen's Cor. 5.1.4 on `φ`, inverted for `θ` at the stored `δ`; `fit` is a
//! maximum-likelihood fit of `(θ, δ)`.
//!
//! References: Joe, H. (1997), *Multivariate Models and Dependence Concepts*,
//! Chapman & Hall, §5.2 (family BB7); Joe, H. (2014), *Dependence Modeling
//! with Copulas*, CRC Press, §4.13.2.
//! Reference: Nelsen, R.B. (2006), "An Introduction to Copulas", 2nd ed., Springer, Corollary 5.1.4 (Kendall's τ `1 + 4∫₀¹ φ/φ′` of an Archimedean copula).

use std::num::NonZeroUsize;
use std::sync::OnceLock;

use gauss_quad::GaussLegendre;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use stochastic_rs_distributions::special::digamma;

use super::BivariateFamily;
use super::conditional::conditional_cdf;
use super::conditional::conditional_quantiles;
use super::two_parameter::clip;
use super::two_parameter::fit_two_parameters;
use super::two_parameter::invert_h;
use super::two_parameter::kendall_tau_numeric;
use super::two_parameter::ln_one_minus_exp;
use crate::bivariate::conditional::check_pairs;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::admitted;
use crate::traits::bivariate::invert_tau;
use crate::traits::bivariate::tau_out_of_range;

/// `ψ(x) − ψ(y)` through `ψ(z) = ψ(z + 1) − 1/z` up past 30, where the asymptotic series of `digamma` holds its digits;
/// Neumaier's compensated sum keeps the thirty-odd recurrence terms from piling up rounding.
fn digamma_difference(mut x: f64, mut y: f64) -> f64 {
  let (mut sum, mut compensation) = (0.0_f64, 0.0_f64);
  let mut add = |v: f64| {
    let t = sum + v;
    compensation += if sum.abs() >= v.abs() {
      (sum - t) + v
    } else {
      (v - t) + sum
    };
    sum = t;
  };
  while x < 30.0 || y < 30.0 {
    add(1.0 / y);
    add(-1.0 / x);
    x += 1.0;
    y += 1.0;
  }
  add(digamma(x));
  add(-digamma(y));
  sum + compensation
}

#[derive(Debug, Clone)]
pub struct Bb7 {
  theta: Option<f64>,
  delta: f64,
}

impl Bb7 {
  /// A BB7 copula; panics unless `1 <= theta < ∞` and `0 < delta < ∞`.
  pub fn new(theta: f64, delta: f64) -> Self {
    Self::try_new(theta, delta).unwrap_or_else(|e| panic!("{e}"))
  }

  /// `new` with the domain error returned instead of a panic.
  pub fn try_new(theta: f64, delta: f64) -> Result<Self, CopulaError> {
    Self::check_delta(delta)?;
    let copula = Self {
      theta: Some(theta),
      delta,
    };
    copula.check_theta()?;
    Ok(copula)
  }

  /// The BB7 copula whose Kendall's τ is `tau` at `delta`.
  pub fn from_tau(tau: f64, delta: f64) -> Result<Self, CopulaError> {
    Self::check_delta(delta)?;
    let unfitted = Self { theta: None, delta };
    Self::try_new(unfitted.theta_from_tau(tau)?, delta)
  }

  pub fn delta(&self) -> f64 {
    self.delta
  }

  /// Panics unless `0 < delta < ∞`.
  pub fn set_delta(&mut self, delta: f64) {
    Self::check_delta(delta).unwrap_or_else(|e| panic!("{e}"));
    self.delta = delta;
  }

  fn check_delta(delta: f64) -> Result<(), CopulaError> {
    if delta.is_finite() && delta > 0.0 {
      return Ok(());
    }
    Err(CopulaError::InvalidParameter {
      name: "delta",
      value: delta,
      constraint: "0.0 < delta < inf".into(),
    })
  }

  fn params(&self) -> Result<(f64, f64), CopulaError> {
    self.check_fit()?;
    Ok((self.theta.expect("checked"), self.delta))
  }

  /// `(ln a, ln b, ln s, ln T, ln(1 − T))` for `a = 1 − ū^θ`, `b = 1 − v̄^θ`,
  /// `s = a^{−δ} + b^{−δ} − 1` and `T = s^{−1/δ}`, with `ū = 1 − u` and
  /// `v̄ = 1 − v`, in the `ln1p`/`expm1` forms that keep the digits when
  /// `ū^θ` is far below the rounding unit (large θ, or u close to one),
  /// where `1 − ū^θ` would round to one and `s` to exactly one.
  fn logs(u: f64, v: f64, theta: f64, delta: f64) -> (f64, f64, f64, f64, f64) {
    let ln_a = ln_one_minus_exp(theta * (-u).ln_1p());
    let ln_b = ln_one_minus_exp(theta * (-v).ln_1p());
    let s_minus_one = (-delta * ln_a).exp_m1() + (-delta * ln_b).exp_m1();
    let ln_s = s_minus_one.ln_1p();
    let ln_t = -ln_s / delta;
    (ln_a, ln_b, ln_s, ln_t, ln_one_minus_exp(ln_t))
  }

  pub(crate) fn cdf_scalar(u: f64, v: f64, theta: f64, delta: f64) -> f64 {
    let (.., ln_1mt) = Self::logs(u, v, theta, delta);
    -(ln_1mt / theta).exp_m1()
  }

  pub(crate) fn h_scalar(u: f64, v: f64, theta: f64, delta: f64) -> f64 {
    let (_, ln_b, ln_s, _, ln_1mt) = Self::logs(u, v, theta, delta);
    ((1.0 / theta - 1.0) * ln_1mt - (1.0 / delta + 1.0) * ln_s - (delta + 1.0) * ln_b
      + (theta - 1.0) * (-v).ln_1p())
    .exp()
  }

  pub(crate) fn log_density_scalar(u: f64, v: f64, theta: f64, delta: f64) -> f64 {
    let (ln_a, ln_b, ln_s, ln_t, ln_1mt) = Self::logs(u, v, theta, delta);
    (1.0 / theta - 2.0) * ln_1mt - (1.0 / delta + 2.0) * ln_s - (delta + 1.0) * (ln_a + ln_b)
      + (theta - 1.0) * ((-u).ln_1p() + (-v).ln_1p())
      + (theta * (delta + 1.0) - (theta * delta + 1.0) * ln_t.exp()).ln()
  }

  /// `ln[Γ(1 + b)Γ(δ + 2)/Γ(δ + 2 + b)] = ∫₀^b ψ(1 + s) − ψ(δ + 2 + s) ds` by 12-point Gauss–Legendre, with its mean over
  /// `[0, b]`, which stays exact at `b = 0`; the pole of `ψ(1 + s)` at `s = −1` keeps `b ≥ −1/2` accurate.
  fn ln_gamma_ratio(b: f64, delta: f64) -> (f64, f64) {
    static RULE: OnceLock<GaussLegendre> = OnceLock::new();
    let rule = RULE.get_or_init(|| GaussLegendre::new(NonZeroUsize::new(12).expect("12 > 0")));
    let g = |s: f64| digamma_difference(1.0 + s, delta + 2.0 + s);
    if b == 0.0 {
      return (0.0, g(0.0));
    }
    let ell = rule.integrate(0.0, b, g);
    (ell, ell / b)
  }

  /// The Nelder–Mead start: bisection of the 100-panel τ at the stored `δ` on `[1, 60]`, 50 steps, `1` when `τ(1) ≥ tau`.
  fn mle_start(&self, tau: f64) -> f64 {
    let kendall_tau =
      |theta: f64| kendall_tau_numeric(|u, v| Self::h_scalar(u, v, theta, self.delta), 100);
    let (mut lo, mut hi) = (1.0_f64, 60.0_f64);
    if kendall_tau(lo) >= tau {
      return lo;
    }
    for _ in 0..50 {
      let mid = 0.5 * (lo + hi);
      if kendall_tau(mid) < tau {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    0.5 * (lo + hi)
  }
}

impl Bb7 {
  /// Maximum-likelihood fit of `(θ, δ)` from a Kendall start without `fit`'s input check, for the vine fitter's
  /// pseudo-observations; a `τ̂` outside BB7's `(0, 1)` or an estimate `try_new` rejects is an error and stores nothing.
  pub(crate) fn fit_parameters(&mut self, X: ArrayView2<f64>) -> Result<(), CopulaError> {
    if X.nrows() < 2 {
      return Err(CopulaError::InsufficientData {
        needed: 2,
        got: X.nrows(),
      });
    }
    let u = X.column(0).to_owned();
    let v = X.column(1).to_owned();
    let (tau, ..) = kendalls::tau_b_with_comparator(&u.to_vec(), &v.to_vec(), |a, b| {
      a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater)
    })
    .map_err(|_| CopulaError::InsufficientData {
      needed: 2,
      got: u.len(),
    })?;
    if !(tau > 0.0 && tau < 1.0) {
      return Err(tau_out_of_range(tau, "0.0 < tau < 1.0".to_string()));
    }
    // A starting point for Nelder–Mead, not an estimate.
    let start_theta = self.mle_start(tau.max(0.05)).clamp(1.0, 20.0);
    let (theta, delta) = fit_two_parameters(
      X,
      (1.0, 0.0),
      (start_theta, self.delta),
      Self::log_density_scalar,
    );
    *self = Self::try_new(theta, delta)?;
    Ok(())
  }
}

impl Default for Bb7 {
  fn default() -> Self {
    Self {
      theta: None,
      delta: 1.0,
    }
  }
}

impl BivariateExt for Bb7 {
  fn family(&self) -> BivariateFamily {
    BivariateFamily::Bb7
  }

  fn theta(&self) -> Option<f64> {
    self.theta
  }

  fn set_theta(&mut self, theta: f64) {
    self.validate_theta(theta).unwrap_or_else(|e| panic!("{e}"));
    self.theta = Some(theta);
  }

  fn theta_bounds(&self) -> (f64, f64) {
    (1.0, f64::INFINITY)
  }

  fn invalid_thetas(&self) -> Vec<f64> {
    vec![]
  }

  fn generator(&self, t: ArrayView1<f64>) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    Ok(t.mapv(|x| (1.0 - (1.0 - x).powf(theta)).powf(-delta) - 1.0))
  }

  fn pdf(&self, X: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    Ok(
      X.rows()
        .into_iter()
        .map(|r| Self::log_density_scalar(clip(r[0]), clip(r[1]), theta, delta).exp())
        .collect(),
    )
  }

  fn cdf(&self, X: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    Ok(
      X.rows()
        .into_iter()
        .map(|r| Self::cdf_scalar(clip(r[0]), clip(r[1]), theta, delta))
        .collect(),
    )
  }

  /// `∂_v C(u, v)`, NaN for `v` outside `[0, 1]`.
  fn partial_derivative(&self, X: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    Ok(conditional_cdf(X, |u, v| {
      Self::h_scalar(clip(u), clip(v), theta, delta)
    }))
  }

  /// The h-function inverted by bisection on `[1e-12, 1 − 1e-12]`, NaN for `y` or `v` outside `[0, 1]`.
  fn percent_point(
    &self,
    y: ArrayView1<f64>,
    conditioning: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    conditional_quantiles(y, conditioning, |p, v| {
      invert_h(|u| Self::h_scalar(u, clip(v), theta, delta), p)
    })
  }

  /// `τ = 1 + 4[R(b) − 1/(1 + b)]/(θ²δb)`, `b = 2/θ − 1`, `R(b) = Γ(1 + b)Γ(δ + 2)/Γ(δ + 2 + b)`: Cor. 5.1.4's Beta form
  /// `1 − 2/(δ(2 − θ)) + 4B(δ + 2, b)/(θ²δ)` without its pole at `θ = 2`; 1 above `θ = 1e100`, where `τ` rounds to it.
  fn tau_from_theta(&self, theta: f64) -> f64 {
    if theta > 1e100 {
      return 1.0;
    }
    let delta = self.delta;
    let c = 2.0 / theta;
    let b = c - 1.0;
    let ratio = if b >= -0.5 {
      let (ell, mean) = Self::ln_gamma_ratio(b, delta);
      mean * if ell == 0.0 { 1.0 } else { ell.exp_m1() / ell }
    } else {
      // `Γ(z + 1) = zΓ(z)` moves `b` to `c = 1 + b`, away from the pole of `ψ(1 + s)` at `s = −1`.
      let (ell, _) = Self::ln_gamma_ratio(c, delta);
      (ell + ((delta + 1.0 + c) / c).ln()).exp_m1() / b
    };
    1.0 + 4.0 / (theta * theta * delta) * (theta / 2.0 + ratio)
  }

  /// Brent on `ln θ` over `[1, 1e4]`, the closed form at every step.
  fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError> {
    let (lo, hi) = (0.0_f64, 1e4_f64.ln());
    let (tau_lo, tau_hi) = (self.tau_from_theta(lo.exp()), self.tau_from_theta(hi.exp()));
    if !(tau >= tau_lo && tau < tau_hi) {
      return Err(tau_out_of_range(
        tau,
        format!("{tau_lo:?} <= tau < {tau_hi:?}"),
      ));
    }
    let theta = invert_tau(|x| self.tau_from_theta(x.exp()) - tau, lo, hi, 1e-12)?.exp();
    admitted(self, tau, theta)
  }

  /// Maximum-likelihood fit of `(θ, δ)` on a two-column sample in `[0, 1]`, started from the Kendall inversion at the
  /// current `δ`.
  fn fit(&mut self, X: ArrayView2<f64>) -> Result<(), CopulaError> {
    check_pairs(X)?;
    self.fit_parameters(X)
  }

  /// `λ_U = 2 − 2^{1/θ}`, `λ_L = 2^{−1/δ}` (Joe 1997, §5.2).
  fn tail_dependence(&self) -> Result<TailDependence<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    Ok(TailDependence {
      lower: 2.0_f64.powf(-1.0 / delta),
      upper: 2.0 - 2.0_f64.powf(1.0 / theta),
    })
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  fn unfitted(delta: f64) -> Bb7 {
    let mut c = Bb7::default();
    c.set_delta(delta);
    c
  }

  #[test]
  fn h_function_is_the_v_derivative_of_the_cdf() {
    let c = Bb7::new(1.8, 0.9);
    for (u, v) in [(0.3, 0.6), (0.8, 0.2), (0.5, 0.5)] {
      let h = c.partial_derivative(array![[u, v]].view()).unwrap()[0];
      let eps = 1e-6;
      let up = c.cdf(array![[u, v + eps]].view()).unwrap()[0];
      let dn = c.cdf(array![[u, v - eps]].view()).unwrap()[0];
      assert!(
        (h - (up - dn) / (2.0 * eps)).abs() < 1e-6,
        "h {h} vs fd {}",
        (up - dn) / (2.0 * eps)
      );
    }
  }

  #[test]
  fn density_is_the_mixed_derivative_of_the_cdf() {
    let c = Bb7::new(1.5, 1.2);
    let (u, v, eps) = (0.4, 0.7, 1e-4);
    let f = |a: f64, b: f64| c.cdf(array![[a, b]].view()).unwrap()[0];
    let mixed = (f(u + eps, v + eps) - f(u + eps, v - eps) - f(u - eps, v + eps)
      + f(u - eps, v - eps))
      / (4.0 * eps * eps);
    let density = c.pdf(array![[u, v]].view()).unwrap()[0];
    assert!(
      (density - mixed).abs() < 1e-5,
      "density {density} vs fd {mixed}"
    );
  }

  #[test]
  fn cdf_has_uniform_margins_and_nests_the_limits() {
    let c = Bb7::new(1.7, 0.8);
    for (u, v) in [(0.2, 0.7), (0.9, 0.4)] {
      assert!((c.cdf(array![[u, 1.0 - 1e-12]].view()).unwrap()[0] - u).abs() < 1e-8);
      assert!((c.cdf(array![[1.0 - 1e-12, v]].view()).unwrap()[0] - v).abs() < 1e-8);
    }
    // θ = 1 is Clayton(δ).
    let mut clayton = crate::bivariate::clayton::Clayton::default();
    clayton.set_theta(0.8);
    let joe_clayton = Bb7::new(1.0, 0.8);
    let (u, v) = (0.35, 0.65);
    assert!(
      (joe_clayton.cdf(array![[u, v]].view()).unwrap()[0]
        - clayton.cdf(array![[u, v]].view()).unwrap()[0])
        .abs()
        < 1e-12
    );
  }

  #[test]
  fn tail_coefficients_follow_joe() {
    let td = Bb7::new(2.0, 1.0).tail_dependence().unwrap();
    assert!((td.upper - (2.0 - 2.0_f64.sqrt())).abs() < 1e-15);
    assert!((td.lower - 0.5).abs() < 1e-15);
  }

  /// Cor. 5.1.4's Beta form by mpmath at 60 digits across its pole at `θ = 2`, at `θ = 1` (Clayton), past the switch at
  /// `θ = 4` and far out; within `1e-15/min(δ, 1)`, as `R(b) − 1/(1 + b)` cancels to `O(δ)`.
  #[test]
  fn kendall_tau_matches_the_beta_form() {
    for (theta, delta, want) in [
      (1.6, 1.1, 0.4577513242482392),
      (1.7, 0.9, 0.4463258932897074),
      (2.0, 1.3, 0.529201420029881),
      (2.000000001, 1.3, 0.529201420137199),
      (1.999999999, 1.3, 0.5292014199225632),
      (1.0000001, 0.9, 0.310344851967264),
      (1.25, 0.3, 0.2246470101786293),
      (4.0, 5.0, 0.7567099567099567),
      (4.0001, 0.05, 0.6171694905640738),
      (60.0, 2.0, 0.9680063458487572),
      (1e6, 1.0, 0.999998000004),
    ] {
      let tau = unfitted(delta).tau_from_theta(theta);
      let tolerance = 1e-15 / delta.min(1.0);
      assert!(
        (tau - want).abs() < tolerance,
        "(θ, δ) = ({theta}, {delta}): {tau} vs {want}"
      );
    }
    assert!((unfitted(2.0).tau_from_theta(3.0) - 0.65).abs() < 1e-15);
    assert!((unfitted(0.8).tau_from_theta(1.0) - 0.8 / 2.8).abs() < 1e-15);
    assert_eq!(unfitted(1.1).tau_from_theta(f64::INFINITY), 1.0);
  }

  #[test]
  fn kendall_inversion_and_maximum_likelihood_are_consistent() {
    let truth = Bb7::new(1.6, 1.1);
    let tau = truth.tau_from_theta(1.6);
    let inverted = Bb7::from_tau(tau, 1.1).unwrap();
    assert!(
      (inverted.theta().unwrap() - 1.6).abs() < 1e-9,
      "inverted theta {}",
      inverted.theta().unwrap()
    );
    let sample = truth.sample(4000, &Deterministic::new(13)).unwrap();
    let mut fitted = Bb7::default();
    fitted.fit(sample.view()).unwrap();
    assert!(
      (fitted.theta().unwrap() - 1.6).abs() < 0.25,
      "theta {}",
      fitted.theta().unwrap()
    );
    assert!(
      (fitted.delta() - 1.1).abs() < 0.3,
      "delta {}",
      fitted.delta()
    );
  }
}
