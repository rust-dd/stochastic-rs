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
pub struct Bb1 {
  pub r#type: CopulaType,
  /// `θ > 0`.
  pub theta: Option<f64>,
  /// `δ ≥ 1`.
  pub delta: f64,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Bb1 {
  pub fn new(theta: Option<f64>, delta: Option<f64>, tau: Option<f64>) -> Self {
    Self {
      r#type: CopulaType::Bb1,
      theta,
      delta: delta.unwrap_or(1.5),
      tau,
      theta_bounds: (0.0, f64::INFINITY),
      invalid_thetas: vec![0.0],
    }
  }

  /// Sets `δ ≥ 1`.
  pub fn with_delta(mut self, delta: f64) -> Self {
    assert!(delta >= 1.0, "BB1 needs δ ≥ 1");
    self.delta = delta;
    self
  }

  fn params(&self) -> Result<(f64, f64), Box<dyn Error>> {
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

  /// Closed-form Kendall's τ at `(θ, δ)`.
  pub fn kendall_tau(theta: f64, delta: f64) -> f64 {
    1.0 - 2.0 / (delta * (theta + 2.0))
  }
}

impl Bb1 {
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
    let start_theta = self.compute_theta().clamp(0.05, 20.0);
    let (theta, delta) = fit_two_parameters(
      X,
      (0.0, 1.0),
      (start_theta, self.delta),
      Self::log_density_scalar,
    );
    self.theta = Some(theta);
    self.delta = delta;
    self.tau = Some(Self::kendall_tau(theta, delta));
    Ok(())
  }
}

impl Default for Bb1 {
  fn default() -> Self {
    Self::new(None, None, None)
  }
}

impl BivariateExt for Bb1 {
  fn r#type(&self) -> CopulaType {
    self.r#type
  }
  fn tau(&self) -> Option<f64> {
    self
      .tau
      .or_else(|| self.theta.map(|t| Self::kendall_tau(t, self.delta)))
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
    Ok(t.mapv(|x| (x.powf(-theta) - 1.0).powf(delta)))
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
  /// `θ = 2 / (δ(1 − τ)) − 2` at the stored `δ`, floored just above zero.
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
    if tau >= 1.0 {
      return f64::INFINITY;
    }
    (2.0 / (self.delta * (1.0 - tau)) - 2.0).max(1e-6)
  }
  /// Maximum-likelihood fit of `(θ, δ)`, started from the Kendall inversion
  /// at the current `δ`.
  fn fit(&mut self, X: &Array2<f64>) -> Result<(), Box<dyn Error>> {
    self.check_marginal(&X.column(0).to_owned())?;
    self.check_marginal(&X.column(1).to_owned())?;
    self.fit_parameters(X)
  }
  /// `λ_L = 2^{−1/(θδ)}`, `λ_U = 2 − 2^{1/δ}` (Joe 1997, §5.2).
  fn tail_dependence(&self) -> TailDependence<f64> {
    self.assert_theta_valid_for_tail_dependence();
    let theta = self.theta.expect("checked");
    TailDependence {
      lower: 2.0_f64.powf(-1.0 / (theta * self.delta)),
      upper: 2.0 - 2.0_f64.powf(1.0 / self.delta),
    }
  }
}

impl Bb1 {
  /// Numerical Kendall's τ from the h-function, for cross-checking the
  /// closed form (and therefore the conditional CDF).
  pub fn kendall_tau_numeric(&self, panels: usize) -> Result<f64, Box<dyn Error>> {
    let (theta, delta) = self.params()?;
    Ok(kendall_tau_numeric(
      |u, v| Self::h_scalar(u, v, theta, delta),
      panels,
    ))
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  #[test]
  fn numeric_kendall_tau_matches_the_closed_form() {
    let c = Bb1::new(Some(0.8), Some(1.6), None);
    let numeric = c.kendall_tau_numeric(400).unwrap();
    let closed = Bb1::kendall_tau(0.8, 1.6);
    assert!(
      (numeric - closed).abs() < 2e-3,
      "numeric {numeric} vs closed {closed}"
    );
    assert!((c.tau().unwrap() - closed).abs() < 1e-15);
  }

  #[test]
  fn h_function_is_the_v_derivative_of_the_cdf() {
    let c = Bb1::new(Some(1.2), Some(1.4), None);
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
  fn cdf_has_uniform_margins_and_nests_clayton() {
    let c = Bb1::new(Some(1.5), Some(1.0), None);
    let mut clayton = crate::bivariate::clayton::Clayton::new();
    clayton.set_theta(1.5);
    for (u, v) in [(0.2, 0.7), (0.9, 0.4)] {
      let ours = c.cdf(&array![[u, v]]).unwrap()[0];
      let theirs = clayton.cdf(&array![[u, v]]).unwrap()[0];
      assert!((ours - theirs).abs() < 1e-12, "{ours} vs {theirs}");
      assert!((c.cdf(&array![[u, 1.0 - 1e-12]]).unwrap()[0] - u).abs() < 1e-8);
    }
  }

  #[test]
  fn tail_coefficients_follow_joe() {
    let c = Bb1::new(Some(0.5), Some(2.0), None);
    let td = c.tail_dependence();
    assert!((td.lower - 2.0_f64.powf(-1.0)).abs() < 1e-15);
    assert!((td.upper - (2.0 - 2.0_f64.sqrt())).abs() < 1e-15);
  }

  #[test]
  fn maximum_likelihood_recovers_the_parameters() {
    let truth = Bb1::new(Some(0.9), Some(1.7), None);
    let sample = truth.sample_with_seed(4000, 11).unwrap();
    let mut fitted = Bb1::default();
    fitted.fit(&sample).unwrap();
    assert!(
      (fitted.theta.unwrap() - 0.9).abs() < 0.2,
      "theta {}",
      fitted.theta.unwrap()
    );
    assert!((fitted.delta - 1.7).abs() < 0.2, "delta {}", fitted.delta);
  }
}
