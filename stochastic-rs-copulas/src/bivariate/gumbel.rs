//! # Gumbel
//!
//! $$
//! C_\theta(u,v)=\exp\!\left(-\left(({-\ln u})^\theta+({-\ln v})^\theta\right)^{1/\theta}\right),\ \theta\ge1
//! $$
//!

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;

use super::BivariateFamily;
use crate::bivariate::conditional::conditional_cdf;
use crate::bivariate::conditional::conditional_quantiles;
use crate::bivariate::conditional::copula_cdf;
use crate::bivariate::conditional::copula_pdf;
use crate::bivariate::conditional::ln_log_ratio;
use crate::bivariate::conditional::ln_ratio;
use crate::bivariate::conditional::log_score_parts;
use crate::bivariate::conditional::numerical_quantiles;
use crate::bivariate::conditional::on_open_square;
use crate::bivariate::conditional::softplus;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::admitted;
use crate::traits::bivariate::tau_out_of_range;

#[derive(Debug, Clone, Default)]
pub struct Gumbel {
  theta: Option<f64>,
}

impl Gumbel {
  /// A Gumbel copula; panics unless `1 <= theta < ∞`.
  pub fn new(theta: f64) -> Self {
    Self::try_new(theta).unwrap_or_else(|e| panic!("{e}"))
  }

  /// `new` with the domain error returned instead of a panic.
  pub fn try_new(theta: f64) -> Result<Self, CopulaError> {
    let copula = Self { theta: Some(theta) };
    copula.check_theta()?;
    Ok(copula)
  }

  /// The copula whose Kendall's τ is `tau`.
  pub fn from_tau(tau: f64) -> Result<Self, CopulaError> {
    Self::try_new(Self::default().theta_from_tau(tau)?)
  }

  /// `ln ∂_v C = (w − m) − m expm1(L/θ) − (1 − 1/θ)L + (θ − 1) ln(w/m)`, `x, w = −ln u, −ln v`, `m = max(x, w)`,
  /// `L = ln(1 + (min/m)^θ)`, the ratio's logarithm by `ln_log_ratio`; the limit laws `δ₀`, `δ₁` at `v = 0, 1`.
  pub(crate) fn ln_h(theta: f64) -> impl Fn(f64, f64) -> f64 {
    move |u, v| {
      if u >= 1.0 || v <= 0.0 {
        return 0.0;
      }
      if u <= 0.0 || v >= 1.0 {
        return f64::NEG_INFINITY;
      }
      let (x, w) = (-u.ln(), -v.ln());
      let q = ln_log_ratio(u, v, x, w);
      let (m, w_minus_m, ln_w_over_m) = if q > 0.0 {
        (x, ln_ratio(u, v), -q)
      } else {
        (w, 0.0, 0.0)
      };
      let log_1p_r = (-theta * q.abs()).exp().ln_1p();
      w_minus_m - m * (log_1p_r / theta).exp_m1() - (1.0 - 1.0 / theta) * log_1p_r
        + (theta - 1.0) * ln_w_over_m
    }
  }

  /// `ln c = m − M expm1(ℓ/θ) + (θ − 1) ln(m/M) − 2(1 − 1/θ)ℓ + ln(1 + (θ − 1)/W)`, `W = M e^{ℓ/θ}`: the `θ`-sized parts of
  /// `(θ − 1) ln(xy) + (1 − 2θ) ln W` cancelled exactly, the last term as a softplus of logarithms.
  fn ln_density(theta: f64, u: f64, v: f64) -> f64 {
    if theta == 1.0 {
      return 0.0;
    }
    let (small, big, ln_ratio, ell) = log_score_parts(theta, u, v);
    let ln_w = big.ln() + ell / theta;
    small - big * (ell / theta).exp_m1() + (theta - 1.0) * ln_ratio
      - 2.0 * (1.0 - 1.0 / theta) * ell
      + softplus((theta - 1.0).ln() - ln_w)
  }
}

impl BivariateExt for Gumbel {
  fn family(&self) -> BivariateFamily {
    BivariateFamily::Gumbel
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
    vec![0.0]
  }

  fn generator(&self, t: ArrayView1<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    Ok((-t.ln()).powf(self.theta.unwrap()))
  }

  fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    copula_pdf(x, |u, v| Self::ln_density(theta, u, v))
  }

  fn log_pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    on_open_square(x, |u, v| Self::ln_density(theta, u, v))
  }

  /// `C = min(u, v) e^{−M expm1(ℓ/θ)}`, the pieces of `Gumbel::ln_density`, as `e^{−M} = min(u, v)`; `uv` at `θ = 1`.
  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    if theta == 1.0 {
      return copula_cdf(x, |u, v| u * v);
    }
    copula_cdf(x, |u, v| {
      let (_, big, _, ell) = log_score_parts(theta, u, v);
      u.min(v) * (-big * (ell / theta).exp_m1()).exp()
    })
  }

  /// `numerical_quantiles` of `Gumbel::ln_h`, the identity at `θ = 1`.
  fn percent_point_unchecked(
    &self,
    y: ArrayView1<f64>,
    v: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    let theta = self.theta.unwrap();
    if theta == 1.0 {
      return Ok(conditional_quantiles(y, v, |y, _| y));
    }
    numerical_quantiles(y, v, self.family(), Self::ln_h(theta))
  }

  /// `exp` of `Gumbel::ln_h`, `u` at `θ = 1`.
  fn partial_derivative(&self, uv: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    if theta == 1.0 {
      return conditional_cdf(uv, |u, _| u);
    }
    let ln_h = Self::ln_h(theta);
    conditional_cdf(uv, |u, v| ln_h(u, v).exp())
  }

  fn tau_from_theta(&self, theta: f64) -> f64 {
    1.0 - 1.0 / theta
  }

  fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError> {
    if !(0.0..1.0).contains(&tau) {
      return Err(tau_out_of_range(tau, "0.0 <= tau < 1.0".into()));
    }
    admitted(self, tau, 1.0 / (1.0 - tau))
  }

  /// Upper-tail dependence $\lambda_U = 2 - 2^{1/\theta}$; Gumbel has no
  /// lower-tail dependence ($\lambda_L = 0$).
  /// Reference: Nelsen, R.B. (2006), "An Introduction to Copulas", 2nd ed.,
  /// Springer, Table 5.1.
  fn tail_dependence(&self) -> Result<TailDependence<f64>, CopulaError> {
    self.check_theta()?;
    let theta = self.theta.unwrap();
    Ok(TailDependence {
      lower: 0.0,
      upper: 2.0 - 2.0_f64.powf(1.0 / theta),
    })
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// Direct regression pin for the `θ=1` `partial_derivative` bug: at
  /// independence (`C(u,v)=uv ⟹ ∂_v C = u`), mirrors
  /// `frank::tests::frank_partial_derivative_at_independence_returns_u`.
  /// This is the test that actually exercises the bug fixed above — see
  /// the next test's doc for why a Kendall's-τ sampling check does not.
  #[test]
  fn gumbel_partial_derivative_at_independence_returns_u() {
    let g = Gumbel::new(1.0);
    let pd = g
      .partial_derivative(array![[0.3_f64, 0.6], [0.2, 0.9]].view())
      .unwrap();
    assert!((pd[0] - 0.3).abs() < 1e-12, "got {}", pd[0]);
    assert!((pd[1] - 0.2).abs() < 1e-12, "got {}", pd[1]);
  }

  /// Kendall's-τ sampling check at `θ=1`, same construction as
  /// `clayton::tests::clayton_independence_sample_kendall_tau_near_zero`
  /// and `frank::tests::frank_independence_sample_kendall_tau_near_zero`.
  /// Unlike those two, this one would pass even without the
  /// `partial_derivative` fix above: `Gumbel::percent_point` already had
  /// its own correct `θ=1` branch, so `Gumbel::sample` never routed
  /// through the buggy `partial_derivative` at this boundary. Kept
  /// anyway as a direct regression guard on `sample`/`percent_point`
  /// itself, matching this crate's standing convention of pinning
  /// independence-limit sampling behavior for every family that has one.
  #[test]
  fn gumbel_independence_sample_kendall_tau_near_zero() {
    let n = 4000usize;
    let se = (2.0 * (2.0 * n as f64 + 5.0) / (9.0 * n as f64 * (n as f64 - 1.0))).sqrt();
    let tol = 6.0 * se;
    let g = Gumbel::new(1.0);
    let best_abs_tau = [2718_u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let samples = g.sample(n, &Deterministic::new(seed)).unwrap();
        let u = samples.column(0).to_vec();
        let v = samples.column(1).to_vec();
        let (tau, ..) =
          kendalls::tau_b_with_comparator(&u, &v, |a: &f64, b: &f64| a.partial_cmp(b).unwrap())
            .unwrap();
        tau.abs()
      })
      .fold(f64::INFINITY, f64::min);
    assert!(
      best_abs_tau < tol,
      "best |tau| across 3 seeds = {best_abs_tau}, expected < {tol} (6 SE, SE={se}, n={n})"
    );
  }

  /// Gumbel θ=2: λ_U = 2 − √2 = 0.585786437626905.
  #[test]
  fn gumbel_tail_dependence_closed_form() {
    let g = Gumbel::new(2.0);
    let td = g.tail_dependence().unwrap();
    assert!(
      (td.upper - (2.0 - std::f64::consts::SQRT_2)).abs() < 1e-12,
      "λ_U={}, expected {}",
      td.upper,
      2.0 - std::f64::consts::SQRT_2
    );
    assert_eq!(td.lower, 0.0);
  }

  /// θ = 0.5 lies below the bound 1, where `λ_U = 2 − 2^{1/θ}` would be negative: no copula holds it.
  #[test]
  fn gumbel_rejects_an_invalid_theta() {
    assert!(matches!(
      Gumbel::try_new(0.5),
      Err(CopulaError::InvalidParameter { name: "theta", .. })
    ));
  }
}
