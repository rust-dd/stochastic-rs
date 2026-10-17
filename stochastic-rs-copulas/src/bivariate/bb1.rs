//! # BB1
//!
//! $$
//! C_{\theta,\delta}(u,v)=\Bigl(1+\bigl[(u^{-\theta}-1)^\delta+(v^{-\theta}-1)^\delta\bigr]^{1/\delta}\Bigr)^{-1/\theta},\qquad \theta>0,\ \delta\ge1
//! $$
//!
//! Joe's BB1 family: the Archimedean copula with generator
//! `φ(t) = (t^{−θ} − 1)^δ`, which nests Clayton (`δ = 1`) and Gumbel
//! (`θ → 0`) and carries both tail coefficients, `λ_L = 2^{−1/(θδ)}` and
//! `λ_U = 2 − 2^{1/δ}`. Kendall's τ is `1 − 2 / (δ(θ + 2))`. With `x = u^{−θ} − 1`,
//! `y = v^{−θ} − 1`, `s = x^δ + y^δ` and `w = s^{1/δ}` the h-function and the
//! density are `∂_v C = (1 + w)^{−1/θ−1} s^{1/δ−1} y^{δ−1} v^{−θ−1}` and
//! `c = (1 + w)^{−1/θ−2} s^{1/δ−2} (xy)^{δ−1} (uv)^{−θ−1} [θ(δ − 1) + (θδ + 1) w]`.
//! `fit` is a maximum-likelihood fit of the pair `(θ, δ)`; the one-parameter
//! Kendall inversion of the trait keeps `δ` fixed and solves `θ` from τ.
//!
//! References: Joe, H. (1997), *Multivariate Models and Dependence Concepts*,
//! Chapman & Hall, §5.2 (family BB1); Joe, H. (2014), *Dependence Modeling
//! with Copulas*, CRC Press, §4.13.1.

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;

use super::BivariateFamily;
use super::conditional::conditional_cdf;
use super::conditional::conditional_quantiles;
use super::two_parameter::clip;
use super::two_parameter::fit_two_parameters;
use super::two_parameter::invert_h;
use super::two_parameter::ln_one_minus_exp;
use crate::bivariate::conditional::check_pairs;
use crate::correlation::kendall_tau_b;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::admitted;
use crate::traits::bivariate::tau_out_of_range;

#[derive(Debug, Clone)]
pub struct Bb1 {
  theta: Option<f64>,
  delta: f64,
}

impl Bb1 {
  /// A BB1 copula; panics unless `0 < theta < ∞` and `1 <= delta < ∞`.
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

  /// The BB1 copula whose Kendall's τ is `tau` at `delta`.
  pub fn from_tau(tau: f64, delta: f64) -> Result<Self, CopulaError> {
    Self::check_delta(delta)?;
    let unfitted = Self { theta: None, delta };
    Self::try_new(unfitted.theta_from_tau(tau)?, delta)
  }

  pub fn delta(&self) -> f64 {
    self.delta
  }

  /// Panics unless `1 <= delta < ∞`.
  pub fn set_delta(&mut self, delta: f64) {
    Self::check_delta(delta).unwrap_or_else(|e| panic!("{e}"));
    self.delta = delta;
  }

  pub(crate) fn check_delta(delta: f64) -> Result<(), CopulaError> {
    if delta.is_finite() && delta >= 1.0 {
      return Ok(());
    }
    Err(CopulaError::InvalidParameter {
      name: "delta",
      value: delta,
      constraint: "1.0 <= delta < inf".into(),
    })
  }

  fn params(&self) -> Result<(f64, f64), CopulaError> {
    self.check_fit()?;
    Ok((self.theta.expect("checked"), self.delta))
  }

  /// `(ln x, ln y, ln s, ln w, ln(1 + w))` for `x = u^{−θ} − 1`,
  /// `y = v^{−θ} − 1`, `s = x^δ + y^δ` and `w = s^{1/δ}`, evaluated in
  /// logarithms so that neither the `u → 0` overflow of `u^{−θ}` nor the
  /// `u → 1` cancellation in `u^{−θ} − 1` costs digits.
  fn logs(u: f64, v: f64, theta: f64, delta: f64) -> (f64, f64, f64, f64, f64) {
    // ln(u^{−θ} − 1) = −θ ln u + ln(1 − u^θ)
    let ln_x = -theta * u.ln() + ln_one_minus_exp(theta * u.ln());
    let ln_y = -theta * v.ln() + ln_one_minus_exp(theta * v.ln());
    let (p, q) = (delta * ln_x, delta * ln_y);
    let m = p.max(q);
    let ln_s = m + ((p - m).exp() + (q - m).exp()).ln();
    let ln_w = ln_s / delta;
    let ln_1pw = if ln_w > 36.0 {
      ln_w
    } else {
      ln_w.exp().ln_1p()
    };
    (ln_x, ln_y, ln_s, ln_w, ln_1pw)
  }

  pub(crate) fn cdf_scalar(u: f64, v: f64, theta: f64, delta: f64) -> f64 {
    let (.., ln_1pw) = Self::logs(u, v, theta, delta);
    (-ln_1pw / theta).exp()
  }

  pub(crate) fn h_scalar(u: f64, v: f64, theta: f64, delta: f64) -> f64 {
    let (_, ln_y, ln_s, _, ln_1pw) = Self::logs(u, v, theta, delta);
    ((-1.0 / theta - 1.0) * ln_1pw + (1.0 / delta - 1.0) * ln_s + (delta - 1.0) * ln_y
      - (theta + 1.0) * v.ln())
    .exp()
  }

  pub(crate) fn log_density_scalar(u: f64, v: f64, theta: f64, delta: f64) -> f64 {
    let (ln_x, ln_y, ln_s, ln_w, ln_1pw) = Self::logs(u, v, theta, delta);
    // ln(θ(δ − 1) + (θδ + 1) w), factored so that neither `w → 0` nor
    // `w → ∞` overflows.
    let bracket = if ln_w > 0.0 {
      ln_w + (theta * delta + 1.0 + theta * (delta - 1.0) * (-ln_w).exp()).ln()
    } else {
      (theta * (delta - 1.0) + (theta * delta + 1.0) * ln_w.exp()).ln()
    };
    (-1.0 / theta - 2.0) * ln_1pw + (1.0 / delta - 2.0) * ln_s + (delta - 1.0) * (ln_x + ln_y)
      - (theta + 1.0) * (u.ln() + v.ln())
      + bracket
  }
}

impl Bb1 {
  /// Maximum-likelihood fit of `(θ, δ)` from a Kendall start without `fit`'s input check, for the vine fitter's
  /// pseudo-observations; a `τ̂` outside BB1's `(0, 1)` or an estimate `try_new` rejects is an error and stores nothing.
  pub(crate) fn fit_parameters(&mut self, x: ArrayView2<f64>) -> Result<(), CopulaError> {
    let tau = kendall_tau_b(x.column(0), x.column(1))?;
    if !(tau > 0.0 && tau < 1.0) {
      return Err(tau_out_of_range(tau, "0.0 < tau < 1.0".to_string()));
    }
    // A starting point for Nelder–Mead, not an estimate.
    let start_theta = (2.0 / (self.delta * (1.0 - tau.max(0.05))) - 2.0).clamp(0.05, 20.0);
    let (theta, delta) = fit_two_parameters(
      x,
      (0.0, 1.0),
      (start_theta, self.delta),
      Self::log_density_scalar,
    );
    *self = Self::try_new(theta, delta)?;
    Ok(())
  }
}

impl Default for Bb1 {
  fn default() -> Self {
    Self {
      theta: None,
      delta: 1.5,
    }
  }
}

impl BivariateExt for Bb1 {
  fn family(&self) -> BivariateFamily {
    BivariateFamily::Bb1
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
    vec![0.0]
  }

  fn generator(&self, t: ArrayView1<f64>) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    Ok(t.mapv(|x| (x.powf(-theta) - 1.0).powf(delta)))
  }

  fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    Ok(
      x.rows()
        .into_iter()
        .map(|r| Self::log_density_scalar(clip(r[0]), clip(r[1]), theta, delta).exp())
        .collect(),
    )
  }

  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    Ok(
      x.rows()
        .into_iter()
        .map(|r| Self::cdf_scalar(clip(r[0]), clip(r[1]), theta, delta))
        .collect(),
    )
  }

  /// `∂_v C(u, v)`, NaN for `v` outside `[0, 1]`.
  fn partial_derivative(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    Ok(conditional_cdf(x, |u, v| {
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

  /// `τ = 1 − 2/(δ(θ + 2))` at the stored `δ` (Joe 1997, §5.2).
  fn tau_from_theta(&self, theta: f64) -> f64 {
    1.0 - 2.0 / (self.delta * (theta + 2.0))
  }

  fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError> {
    let lo = 1.0 - 1.0 / self.delta;
    if !(tau > lo && tau < 1.0) {
      return Err(tau_out_of_range(tau, format!("{lo:?} < tau < 1.0")));
    }
    // Within an ulp of the lower end the difference can round to `θ ≤ 0`, outside the family.
    admitted(self, tau, 2.0 / (self.delta * (1.0 - tau)) - 2.0)
  }

  /// Maximum-likelihood fit of `(θ, δ)` on a two-column sample in `[0, 1]`, started from the Kendall inversion at the
  /// current `δ`.
  fn fit(&mut self, x: ArrayView2<f64>) -> Result<(), CopulaError> {
    check_pairs(x)?;
    self.fit_parameters(x)
  }

  /// `λ_L = 2^{−1/(θδ)}`, `λ_U = 2 − 2^{1/δ}` (Joe 1997, §5.2).
  fn tail_dependence(&self) -> Result<TailDependence<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    Ok(TailDependence {
      lower: 2.0_f64.powf(-1.0 / (theta * delta)),
      upper: 2.0 - 2.0_f64.powf(1.0 / delta),
    })
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::bivariate::two_parameter::kendall_tau_numeric;

  /// Kendall's τ from the h-function alone, which cross-checks the closed form and with it the conditional cdf.
  fn kendall_tau_from_h(c: &Bb1, panels: usize) -> f64 {
    let (theta, delta) = c.params().unwrap();
    kendall_tau_numeric(|u, v| Bb1::h_scalar(u, v, theta, delta), panels)
  }

  #[test]
  fn numeric_kendall_tau_matches_the_closed_form() {
    let c = Bb1::new(0.8, 1.6);
    let numeric = kendall_tau_from_h(&c, 400);
    let closed = c.tau_from_theta(0.8);
    assert!(
      (numeric - closed).abs() < 2e-3,
      "numeric {numeric} vs closed {closed}"
    );
    assert_eq!(c.tau().unwrap(), closed);
  }

  #[test]
  fn h_function_is_the_v_derivative_of_the_cdf() {
    let c = Bb1::new(1.2, 1.4);
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
  fn cdf_has_uniform_margins_and_nests_clayton() {
    let c = Bb1::new(1.5, 1.0);
    let mut clayton = crate::bivariate::clayton::Clayton::default();
    clayton.set_theta(1.5);
    for (u, v) in [(0.2, 0.7), (0.9, 0.4)] {
      let ours = c.cdf(array![[u, v]].view()).unwrap()[0];
      let theirs = clayton.cdf(array![[u, v]].view()).unwrap()[0];
      assert!((ours - theirs).abs() < 1e-12, "{ours} vs {theirs}");
      assert!((c.cdf(array![[u, 1.0 - 1e-12]].view()).unwrap()[0] - u).abs() < 1e-8);
    }
  }

  #[test]
  fn tail_coefficients_follow_joe() {
    let c = Bb1::new(0.5, 2.0);
    let td = c.tail_dependence().unwrap();
    assert!((td.lower - 2.0_f64.powf(-1.0)).abs() < 1e-15);
    assert!((td.upper - (2.0 - 2.0_f64.sqrt())).abs() < 1e-15);
  }

  #[test]
  fn maximum_likelihood_recovers_the_parameters() {
    let truth = Bb1::new(0.9, 1.7);
    let sample = truth.sample(4000, &Deterministic::new(11)).unwrap();
    let mut fitted = Bb1::default();
    fitted.fit(sample.view()).unwrap();
    assert!(
      (fitted.theta().unwrap() - 0.9).abs() < 0.2,
      "theta {}",
      fitted.theta().unwrap()
    );
    assert!(
      (fitted.delta() - 1.7).abs() < 0.2,
      "delta {}",
      fitted.delta()
    );
  }
}
