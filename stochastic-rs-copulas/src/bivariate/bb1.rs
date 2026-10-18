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
use stochastic_rs_distributions::RealExt;

use super::BivariateFamily;
use super::conditional::conditional_cdf;
use super::conditional::copula_cdf;
use super::conditional::copula_pdf;
use super::conditional::expm1_over;
use super::conditional::ln_1p_over;
use super::conditional::ln_add_exp;
use super::conditional::ln_expm1;
use super::conditional::ln_one_minus_exp;
use super::conditional::ln_one_minus_pow;
use super::conditional::ln_one_minus_pow_ratio;
use super::conditional::ln_ratio;
use super::conditional::numerical_quantiles;
use super::conditional::on_open_square;
use super::conditional::softplus;
use super::two_parameter::fit_two_parameters;
use crate::bivariate::conditional::check_open;
use crate::bivariate::conditional::check_pairs;
use crate::correlation::kendall_tau_b;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::admitted;
use crate::traits::bivariate::checked_percent_point;
use crate::traits::bivariate::tau_out_of_range;

/// The expanded h exponentiates logarithms that grow with `θ`, `δ` and `|ln u|`; below `2⁷` in magnitude their rounding
/// stays under `2⁻⁴²` relative.
const EXPANDED_LIMIT: f64 = 128.0;

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

  /// `ln(X/Y)`, `X, Y = u^{−θ} − 1, v^{−θ} − 1`, as `θ ln(v/u) + ln((1 − u^θ)/(1 − v^θ))`, exact near a ratio of 1.
  fn ln_generator_ratio(u: f64, v: f64, ln_u: f64, ln_v: f64, theta: f64) -> f64 {
    theta * ln_ratio(v, u) + ln_one_minus_pow_ratio(theta, ln_u, ln_v)
  }

  /// `(W, P, ln r, ℓ)` while `θ max(x, y) ≤ 700`, `x, y = −ln u, −ln v`: `P = X_max/θ = x_max expm1(θx_max)/(θx_max)`,
  /// `r = X_min/X_max`, `ℓ = ln(1 + r^δ)`, `W = w/θ = Pe^{ℓ/δ}`, so a small `θ` keeps its digits.
  fn scaled(u: f64, v: f64, ln_u: f64, ln_v: f64, theta: f64, delta: f64) -> (f64, f64, f64, f64) {
    let ln_xy = Self::ln_generator_ratio(u, v, ln_u, ln_v, theta);
    let big = -(if ln_xy >= 0.0 { ln_u } else { ln_v });
    let p_big = big * expm1_over(theta * big);
    let ln_r = -ln_xy.abs();
    let ell = (delta * ln_r).exp().ln_1p();
    (p_big * (ell / delta).exp(), p_big, ln_r, ell)
  }

  /// `(λ_M, λ_m, ln r, ℓ, ln w)` past `θ max(x, y) = 700`, `λ = ln(1 − u^θ)` of the smaller and larger coordinate,
  /// `ln w = θM + λ_M + ℓ/δ` with `M = −ln min(u, v)`.
  fn recentred(
    u: f64,
    v: f64,
    ln_u: f64,
    ln_v: f64,
    theta: f64,
    delta: f64,
  ) -> (f64, f64, f64, f64, f64) {
    let (ln_low, ln_high) = if u <= v { (ln_u, ln_v) } else { (ln_v, ln_u) };
    let (lam_big, lam_small) = (
      ln_one_minus_pow(theta, ln_low),
      ln_one_minus_pow(theta, ln_high),
    );
    let ln_r = -Self::ln_generator_ratio(u, v, ln_u, ln_v, theta).abs();
    let ell = (delta * ln_r).exp().ln_1p();
    (
      lam_big,
      lam_small,
      ln_r,
      ell,
      -theta * ln_low + lam_big + ell / delta,
    )
  }

  /// `C = (1 + w)^{−1/θ}`: `e^{−W ln(1 + θW)/(θW)}` by `Bb1::scaled`, past it `min(u, v) e^{−(λ_M + ℓ/δ + ln(1 + 1/w))/θ}`.
  pub(crate) fn cdf_scalar(u: f64, v: f64, theta: f64, delta: f64) -> f64 {
    let (ln_u, ln_v) = (u.ln(), v.ln());
    if -theta * ln_u.min(ln_v) <= 700.0 {
      let (w_over, ..) = Self::scaled(u, v, ln_u, ln_v, theta, delta);
      return (-w_over * ln_1p_over(theta * w_over)).exp();
    }
    let (lam_big, _, _, ell, ln_w) = Self::recentred(u, v, ln_u, ln_v, theta, delta);
    u.min(v) * (-(lam_big + ell / delta + (-ln_w).exp().ln_1p()) / theta).exp()
  }

  /// `∂_v C`, held below 1 against rounding: `ln_h_interior` exponentiated, and the limit laws at `v = 0` (`δ₀`) and
  /// `v = 1` (`u^{θ+1}` at `δ = 1`, else `δ₁`).
  pub(crate) fn h_scalar(u: f64, v: f64, theta: f64, delta: f64) -> f64 {
    if u >= 1.0 || v <= 0.0 {
      return 1.0;
    }
    if u <= 0.0 {
      return 0.0;
    }
    if v >= 1.0 {
      return if delta == 1.0 {
        u.powf(theta + 1.0)
      } else {
        0.0
      };
    }
    Self::ln_h_interior(u, v, theta, delta)
      .exp()
      .min_or_nan(1.0)
  }

  /// `(ln X, ln Y, ln s, ln(1 + w))` of the expanded h, `s = X^δ + Y^δ`, `w = s^{1/δ}`, each from logarithms.
  fn expanded_logs(u: f64, v: f64, theta: f64, delta: f64) -> (f64, f64, f64, f64) {
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
    (ln_x, ln_y, ln_s, ln_1pw)
  }

  /// `ln ∂_v C` on the open square: the expanded form while `ln x`, `ln y` and its exponent terms stay below
  /// `EXPANDED_LIMIT`, `ln_h_log` past it.
  fn ln_h_interior(u: f64, v: f64, theta: f64, delta: f64) -> f64 {
    let (ln_x, ln_y, ln_s, ln_1pw) = Self::expanded_logs(u, v, theta, delta);
    let terms = [
      (-1.0 / theta - 1.0) * ln_1pw,
      (1.0 / delta - 1.0) * ln_s,
      (delta - 1.0) * ln_y,
      -(theta + 1.0) * v.ln(),
    ];
    let moderate = ln_x.abs().max_or_nan(ln_y.abs()) <= EXPANDED_LIMIT;
    if moderate && terms.iter().all(|t| t.abs() <= EXPANDED_LIMIT) {
      return terms[0] + terms[1] + terms[2] + terms[3];
    }
    Self::ln_h_log(u, v, theta, delta)
  }

  /// `ln ∂_v C` with `h_scalar`'s limit laws at the edges and `ln_h_log` inside, whose `1 − h` keeps its digits near 1:
  /// what `percent_point` inverts.
  fn ln_h(u: f64, v: f64, theta: f64, delta: f64) -> f64 {
    if u >= 1.0 || v <= 0.0 {
      0.0
    } else if u <= 0.0 {
      f64::NEG_INFINITY
    } else if v >= 1.0 {
      if delta == 1.0 {
        (theta + 1.0) * u.ln()
      } else {
        f64::NEG_INFINITY
      }
    } else {
      Self::ln_h_log(u, v, theta, delta)
    }
  }

  /// `ln h = −((δ − 1)/δ)S − (1 + 1/θ) ln(1 + (1 − v^θ) expm1(S/δ))`, `S = ln(1 + (x/y)^δ)`, with
  /// `ln(x/y) = θ ln(v/u) + ln((1 − u^θ)/(1 − v^θ))`, every factor a logarithm, so nothing overflows, cancels or underflows.
  fn ln_h_log(u: f64, v: f64, theta: f64, delta: f64) -> f64 {
    let (ln_u, ln_v) = (u.ln(), v.ln());
    let ln_x_over_y = theta * ln_ratio(v, u) + ln_one_minus_pow_ratio(theta, ln_u, ln_v);
    let s = softplus(delta * ln_x_over_y);
    let clayton = if delta == 1.0 {
      0.0
    } else {
      (delta - 1.0) / delta * s
    };
    let z = s / delta;
    let ln_p = ln_one_minus_pow(theta, ln_v) + ln_expm1(z, s.ln() - delta.ln());
    // `P < 1` as `(1 + θ)((1 − v^θ)/θ) expm1(S/δ) ln(1 + P)/P`: through `ln P` the `1/θ` would scale its rounding.
    let tail = if ln_p < 0.0 {
      let p = ln_p.exp();
      let ln_1p_over = if p == 0.0 { 1.0 } else { p.ln_1p() / p };
      let e = Self::expm1_s_over_delta(ln_u, ln_v, theta, delta, ln_x_over_y, z);
      (1.0 + theta) * (-ln_v * expm1_over(theta * ln_v)) * e * ln_1p_over
    } else if theta < 0.5 {
      // `1 + P = v^θ(1 + w)`: `ln P`'s rounding, up to `1500ε`, scaled by `1 + 1/θ`, past `5e-13` below `θ = 1/2`, where
      // `P ≥ 1` needs `w ≳ 1`, so the generators' terms stay under `2000ε`.
      let (_, _, ln_s, _) = Self::expanded_logs(u, v, theta, delta);
      (theta + 1.0) * ln_v + (1.0 + 1.0 / theta) * softplus(ln_s / delta)
    } else {
      (1.0 + 1.0 / theta) * softplus(ln_p)
    };
    -clayton - tail
  }

  /// `ln c` at one point, for a caller holding no `Bb1Density`.
  pub(crate) fn log_density_scalar(u: f64, v: f64, theta: f64, delta: f64) -> f64 {
    Bb1Density::new(theta, delta).ln_c(u, v)
  }

  /// `numerical_quantiles` of `Bb1::ln_h`.
  fn unchecked_inverse(
    &self,
    y: ArrayView1<f64>,
    conditioning: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    numerical_quantiles(y, conditioning, self.family(), |u, v| {
      Self::ln_h(u, v, theta, delta)
    })
  }
}

/// BB1's log-density at one `(θ, δ)`, with what no point changes formed once.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Bb1Density {
  theta: f64,
  delta: f64,
  ln_theta: f64,
  ln_delta: f64,
}

impl Bb1Density {
  pub(crate) fn new(theta: f64, delta: f64) -> Self {
    Self {
      theta,
      delta,
      ln_theta: theta.ln(),
      ln_delta: delta.ln(),
    }
  }

  /// `ln c` on the open square: `Bb1::scaled`'s form, `ln θ` cancelled, while `θ max(x, y) ≤ 700`, else re-centred on
  /// `−θδ(M − m)`, where the `θ`-sized terms of `−(1/θ + 2) ln(1 + w) + (θ + 1)(x + y) − ln X_max` cancel exactly.
  pub(crate) fn ln_c(&self, u: f64, v: f64) -> f64 {
    let (theta, delta) = (self.theta, self.delta);
    let (ln_u, ln_v) = (u.ln(), v.ln());
    let (x, y) = (-ln_u, -ln_v);
    if theta * x.max(y) <= 700.0 {
      let (w_over, p_big, ln_r, ell) = Bb1::scaled(u, v, ln_u, ln_v, theta, delta);
      let w = theta * w_over;
      let ln_1pw = w.ln_1p();
      let ln_1p_over_w = if w == 0.0 { 1.0 } else { ln_1pw / w };
      return -w_over * ln_1p_over_w - 2.0 * ln_1pw + (theta + 1.0) * (x + y) - p_big.ln()
        + (delta - 1.0) * ln_r
        + (1.0 / delta - 2.0) * ell
        + ln_add_exp((delta - 1.0 + w_over).ln(), self.ln_delta + w.ln());
    }
    let (lam_big, lam_small, _, ell, ln_w) = Bb1::recentred(u, v, ln_u, ln_v, theta, delta);
    let tail = (-ln_w).exp();
    -(theta * ln_ratio(u, v).abs()) * delta + x.min(y) - (2.0 + 1.0 / theta) * lam_big
      + (delta - 1.0) * (lam_small - lam_big)
      - (2.0 + 1.0 / (theta * delta)) * ell
      - (2.0 + 1.0 / theta) * tail.ln_1p()
      + self.ln_theta
      + (delta + 1.0 / theta + (delta - 1.0) * tail).ln()
  }
}

impl Bb1 {
  /// `expm1(S/δ) = (1 + r^δ)^{1/δ} − 1`, `r = x/y`: past `S/δ = 1`, `r e^{ln(1 + r^{−δ})/δ} − 1` with `r` formed directly,
  /// as exponentiating `ln r` would cost `|ln r|` rounding units.
  fn expm1_s_over_delta(ln_u: f64, ln_v: f64, theta: f64, delta: f64, ln_r: f64, z: f64) -> f64 {
    let (t_u, t_v) = (-theta * ln_u, -theta * ln_v);
    let r = if t_u <= 1.0 && t_v <= 1.0 {
      (ln_u / ln_v) * (expm1_over(t_u) / expm1_over(t_v))
    } else {
      t_u.exp_m1() / t_v.exp_m1()
    };
    if z > 1.0 && r.is_finite() && r > 0.0 {
      r * (softplus(-delta * ln_r) / delta).exp() - 1.0
    } else {
      z.exp_m1()
    }
  }

  /// Maximum-likelihood fit of `(θ, δ)` from a Kendall start without `fit`'s input check, for the vine fitter's
  /// pseudo-observations; a `τ̂` outside BB1's `(0, 1)` or an estimate `try_new` rejects is an error and stores nothing.
  pub(crate) fn fit_parameters(
    &mut self,
    u: ArrayView1<f64>,
    v: ArrayView1<f64>,
  ) -> Result<(), CopulaError> {
    let tau = kendall_tau_b(u, v)?;
    if !(tau > 0.0 && tau < 1.0) {
      return Err(tau_out_of_range(tau, "0.0 < tau < 1.0".to_string()));
    }
    // A starting point for Nelder–Mead, not an estimate.
    let start_theta = (2.0 / (self.delta * (1.0 - tau.max(0.05))) - 2.0).clamp(0.05, 20.0);
    let (theta, delta) = fit_two_parameters(
      u,
      v,
      (0.0, 1.0),
      (start_theta, self.delta),
      |theta, delta| {
        let density = Bb1Density::new(theta, delta);
        move |a, b| density.ln_c(a, b)
      },
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

  /// `φ(t) = (t^{−θ} − 1)^δ` as `exp(δ ln expm1(−θ ln t))`, which keeps the digits `t^{−θ} − 1` loses as `θ → 0`.
  fn generator(&self, t: ArrayView1<f64>) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    Ok(t.mapv(|t| {
      let minus_ln_t = -t.ln();
      (delta * ln_expm1(theta * minus_ln_t, theta.ln() + minus_ln_t.ln())).exp()
    }))
  }

  fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    let density = Bb1Density::new(theta, delta);
    copula_pdf(x, |u, v| density.ln_c(u, v))
  }

  fn log_pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    let density = Bb1Density::new(theta, delta);
    on_open_square(x, |u, v| density.ln_c(u, v))
  }

  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    copula_cdf(x, |u, v| Self::cdf_scalar(u, v, theta, delta))
  }

  /// `∂_v C(u, v)`.
  fn partial_derivative(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    conditional_cdf(x, |u, v| Self::h_scalar(u, v, theta, delta))
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
    conditioning: ArrayView1<f64>,
  ) -> Option<Result<Array1<f64>, CopulaError>> {
    Some(self.unchecked_inverse(y, conditioning))
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

  /// Maximum-likelihood fit of `(θ, δ)`, started from the Kendall inversion at the current `δ`.
  fn fit(&mut self, x: ArrayView2<f64>) -> Result<(), CopulaError> {
    check_pairs(x)?;
    check_open(x)?;
    self.fit_parameters(x.column(0), x.column(1))
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
