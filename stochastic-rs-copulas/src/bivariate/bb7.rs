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
//! Kendall's τ has no elementary closed form and is integrated numerically;
//! the Kendall inversion of the trait solves `θ` at the stored `δ` by
//! bisection, and `fit` is a maximum-likelihood fit of `(θ, δ)`.
//!
//! References: Joe, H. (1997), *Multivariate Models and Dependence Concepts*,
//! Chapman & Hall, §5.2 (family BB7); Joe, H. (2014), *Dependence Modeling
//! with Copulas*, CRC Press, §4.13.2.

use std::error::Error;

use ndarray::Array1;
use ndarray::Array2;

use super::CopulaType;
use super::two_parameter::clip;
use super::two_parameter::fit_two_parameters;
use super::two_parameter::invert_h;
use super::two_parameter::kendall_tau_numeric;
use super::two_parameter::ln_one_minus_exp;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;

#[derive(Debug, Clone)]
pub struct Bb7 {
  pub r#type: CopulaType,
  /// `θ ≥ 1`.
  pub theta: Option<f64>,
  /// `δ > 0`.
  pub delta: f64,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Bb7 {
  pub fn new(theta: Option<f64>, delta: Option<f64>, tau: Option<f64>) -> Self {
    Self {
      r#type: CopulaType::Bb7,
      theta,
      delta: delta.unwrap_or(1.0),
      tau,
      theta_bounds: (1.0, f64::INFINITY),
      invalid_thetas: Vec::new(),
    }
  }

  /// Sets `δ > 0`.
  pub fn with_delta(mut self, delta: f64) -> Self {
    assert!(delta > 0.0, "BB7 needs δ > 0");
    self.delta = delta;
    self
  }

  fn params(&self) -> Result<(f64, f64), Box<dyn Error>> {
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

  /// Numerical Kendall's τ at `(θ, δ)` from the h-function (composite
  /// two-point Gauss–Legendre); BB7 has no closed form.
  pub fn kendall_tau(theta: f64, delta: f64, panels: usize) -> f64 {
    kendall_tau_numeric(|u, v| Self::h_scalar(u, v, theta, delta), panels)
  }
}

impl Bb7 {
  /// Maximum-likelihood fit of `(θ, δ)` started from the Kendall inversion,
  /// without the marginal uniformity check that [`BivariateExt::fit`] runs
  /// first — for callers that already hold pseudo-observations, such as the
  /// vine fitter.
  pub(crate) fn fit_parameters(&mut self, X: &Array2<f64>) -> Result<(), Box<dyn Error>> {
    let u = X.column(0).to_owned();
    let v = X.column(1).to_owned();
    let (tau, ..) = kendalls::tau_b_with_comparator(&u.to_vec(), &v.to_vec(), |a, b| {
      a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater)
    })?;
    self.tau = Some(tau.max(0.05));
    let start_theta = self.compute_theta().clamp(1.0, 20.0);
    let (theta, delta) = fit_two_parameters(
      X,
      (1.0, 0.0),
      (start_theta, self.delta),
      Self::log_density_scalar,
    );
    self.theta = Some(theta);
    self.delta = delta;
    self.tau = Some(Self::kendall_tau(theta, delta, 200));
    Ok(())
  }
}

impl Default for Bb7 {
  fn default() -> Self {
    Self::new(None, None, None)
  }
}

impl BivariateExt for Bb7 {
  fn r#type(&self) -> CopulaType {
    self.r#type
  }
  fn tau(&self) -> Option<f64> {
    self
      .tau
      .or_else(|| self.theta.map(|t| Self::kendall_tau(t, self.delta, 200)))
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
  fn generator(&self, t: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    let (theta, delta) = self.params()?;
    Ok(t.mapv(|x| (1.0 - (1.0 - x).powf(theta)).powf(-delta) - 1.0))
  }
  fn pdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    let (theta, delta) = self.params()?;
    Ok(
      X.rows()
        .into_iter()
        .map(|r| Self::log_density_scalar(clip(r[0]), clip(r[1]), theta, delta).exp())
        .collect(),
    )
  }
  fn cdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    let (theta, delta) = self.params()?;
    Ok(
      X.rows()
        .into_iter()
        .map(|r| Self::cdf_scalar(clip(r[0]), clip(r[1]), theta, delta))
        .collect(),
    )
  }
  /// `∂_v C(u, v)`.
  fn partial_derivative(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    let (theta, delta) = self.params()?;
    Ok(
      X.rows()
        .into_iter()
        .map(|r| Self::h_scalar(clip(r[0]), clip(r[1]), theta, delta))
        .collect(),
    )
  }
  /// `θ ∈ [1, 60]` solving the numerical τ at the stored `δ` by bisection.
  /// Inverse h-function by bisection: the trait's default root finder does
  /// not converge on the steep conditional CDF near `v → 1`.
  fn percent_point(
    &self,
    y: &Array1<f64>,
    conditioning: &Array1<f64>,
  ) -> Result<Array1<f64>, Box<dyn Error>> {
    let (theta, delta) = self.params()?;
    Ok(
      y.iter()
        .zip(conditioning)
        .map(|(&p, &v)| invert_h(|u| Self::h_scalar(u, clip(v), theta, delta), p))
        .collect(),
    )
  }
  fn compute_theta(&self) -> f64 {
    let tau = self.tau.expect("set tau first");
    let (mut lo, mut hi) = (1.0_f64, 60.0_f64);
    if Self::kendall_tau(lo, self.delta, 100) >= tau {
      return lo;
    }
    for _ in 0..50 {
      let mid = 0.5 * (lo + hi);
      if Self::kendall_tau(mid, self.delta, 100) < tau {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    0.5 * (lo + hi)
  }
  /// Maximum-likelihood fit of `(θ, δ)`, started from the Kendall inversion
  /// at the current `δ`.
  fn fit(&mut self, X: &Array2<f64>) -> Result<(), Box<dyn Error>> {
    self.check_marginal(&X.column(0).to_owned())?;
    self.check_marginal(&X.column(1).to_owned())?;
    self.fit_parameters(X)
  }
  /// `λ_U = 2 − 2^{1/θ}`, `λ_L = 2^{−1/δ}` (Joe 1997, §5.2).
  fn tail_dependence(&self) -> TailDependence<f64> {
    self.assert_theta_valid_for_tail_dependence();
    let theta = self.theta.expect("checked");
    TailDependence {
      lower: 2.0_f64.powf(-1.0 / self.delta),
      upper: 2.0 - 2.0_f64.powf(1.0 / theta),
    }
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  #[test]
  fn h_function_is_the_v_derivative_of_the_cdf() {
    let c = Bb7::new(Some(1.8), Some(0.9), None);
    for (u, v) in [(0.3, 0.6), (0.8, 0.2), (0.5, 0.5)] {
      let h = c.partial_derivative(&array![[u, v]]).unwrap()[0];
      let eps = 1e-6;
      let up = c.cdf(&array![[u, v + eps]]).unwrap()[0];
      let dn = c.cdf(&array![[u, v - eps]]).unwrap()[0];
      assert!(
        (h - (up - dn) / (2.0 * eps)).abs() < 1e-6,
        "h {h} vs fd {}",
        (up - dn) / (2.0 * eps)
      );
    }
  }

  #[test]
  fn density_is_the_mixed_derivative_of_the_cdf() {
    let c = Bb7::new(Some(1.5), Some(1.2), None);
    let (u, v, eps) = (0.4, 0.7, 1e-4);
    let f = |a: f64, b: f64| c.cdf(&array![[a, b]]).unwrap()[0];
    let mixed = (f(u + eps, v + eps) - f(u + eps, v - eps) - f(u - eps, v + eps)
      + f(u - eps, v - eps))
      / (4.0 * eps * eps);
    let density = c.pdf(&array![[u, v]]).unwrap()[0];
    assert!(
      (density - mixed).abs() < 1e-5,
      "density {density} vs fd {mixed}"
    );
  }

  #[test]
  fn cdf_has_uniform_margins_and_nests_the_limits() {
    let c = Bb7::new(Some(1.7), Some(0.8), None);
    for (u, v) in [(0.2, 0.7), (0.9, 0.4)] {
      assert!((c.cdf(&array![[u, 1.0 - 1e-12]]).unwrap()[0] - u).abs() < 1e-8);
      assert!((c.cdf(&array![[1.0 - 1e-12, v]]).unwrap()[0] - v).abs() < 1e-8);
    }
    // θ = 1 is Clayton(δ).
    let mut clayton = crate::bivariate::clayton::Clayton::new();
    clayton.set_theta(0.8);
    let joe_clayton = Bb7::new(Some(1.0), Some(0.8), None);
    let (u, v) = (0.35, 0.65);
    assert!(
      (joe_clayton.cdf(&array![[u, v]]).unwrap()[0] - clayton.cdf(&array![[u, v]]).unwrap()[0])
        .abs()
        < 1e-12
    );
  }

  #[test]
  fn tail_coefficients_follow_joe() {
    let td = Bb7::new(Some(2.0), Some(1.0), None).tail_dependence();
    assert!((td.upper - (2.0 - 2.0_f64.sqrt())).abs() < 1e-15);
    assert!((td.lower - 0.5).abs() < 1e-15);
  }

  #[test]
  fn kendall_inversion_and_maximum_likelihood_are_consistent() {
    let truth = Bb7::new(Some(1.6), Some(1.1), None);
    let tau = Bb7::kendall_tau(1.6, 1.1, 200);
    let mut inverted = Bb7::new(None, Some(1.1), Some(tau));
    inverted.theta = Some(inverted.compute_theta());
    assert!(
      (inverted.theta.unwrap() - 1.6).abs() < 0.02,
      "inverted theta {}",
      inverted.theta.unwrap()
    );
    let sample = truth.sample_with_seed(4000, 13).unwrap();
    let mut fitted = Bb7::default();
    fitted.fit(&sample).unwrap();
    assert!(
      (fitted.theta.unwrap() - 1.6).abs() < 0.25,
      "theta {}",
      fitted.theta.unwrap()
    );
    assert!((fitted.delta - 1.1).abs() < 0.3, "delta {}", fitted.delta);
  }
}
