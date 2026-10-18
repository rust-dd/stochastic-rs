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
use stochastic_rs_distributions::RealExt;
use stochastic_rs_distributions::special::digamma;

use super::BivariateFamily;
use super::conditional::conditional_cdf;
use super::conditional::copula_cdf;
use super::conditional::copula_log_pdf;
use super::conditional::copula_pdf;
use super::conditional::expm1_over;
use super::conditional::ln_1p_over;
use super::conditional::ln_one_minus_exp;
use super::conditional::ln_one_minus_pow;
use super::conditional::ln_one_minus_pow_ratio;
use super::conditional::ln_ratio_complement;
use super::conditional::numerical_quantiles;
use super::conditional::softplus;
use super::two_parameter::clip;
use super::two_parameter::fit_two_parameters;
use super::two_parameter::kendall_tau_numeric;
use crate::bivariate::conditional::check_pairs;
use crate::correlation::kendall_tau_b;
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

/// `ln(1 − e^{−z})` for `z ≥ 0`, from `ln z` where `z` is subnormal.
fn ln_one_minus_exp_from(z: f64, ln_z: f64) -> f64 {
  if z < f64::MIN_POSITIVE {
    ln_z
  } else {
    ln_one_minus_exp(-z)
  }
}

/// `ln φ(z) = ln((1 − e^{−z})/z)` for `z ≥ 0` given `ln z`: its series `−z/2 + z²/24` below `1e-5`, where the
/// difference of the two logarithms would cancel, and 0 where `z` underflows.
fn ln_phi(z: f64, ln_z: f64) -> f64 {
  if z < 1e-5 {
    z * (z / 24.0 - 0.5)
  } else {
    ln_one_minus_exp(-z) - ln_z
  }
}

/// The expanded h's terms are differences of logarithms that grow with `θ`, `δ` and `|ln v̄|`; below `2⁷` their rounding
/// stays under `2⁻⁴²` relative.
const EXPANDED_LIMIT: f64 = 128.0;

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

  pub(crate) fn check_delta(delta: f64) -> Result<(), CopulaError> {
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

  /// `∂_v C`, held below 1 against rounding: `ln_h_interior` exponentiated, and the limit laws at `v = 0` (`δ₀`) and
  /// `v = 1` (`u^{δ+1}` at `θ = 1`, else `δ₁`).
  pub(crate) fn h_scalar(u: f64, v: f64, theta: f64, delta: f64) -> f64 {
    if u >= 1.0 || v <= 0.0 {
      return 1.0;
    }
    if u <= 0.0 {
      return 0.0;
    }
    if v >= 1.0 {
      return if theta == 1.0 {
        u.powf(delta + 1.0)
      } else {
        0.0
      };
    }
    Self::ln_h_interior(u, v, theta, delta)
      .exp()
      .min_or_nan(1.0)
  }

  /// The four terms of the expanded `ln ∂_v C`, and the `ln s`, `ln T` its range check reads: the one plain kernel,
  /// summed by `ln_h_interior` and by the Kendall quadrature of `mle_start`.
  #[inline(always)]
  fn expanded(u: f64, v: f64, theta: f64, delta: f64) -> ([f64; 4], f64, f64) {
    let (_, ln_b, ln_s, ln_t, ln_1mt) = Self::logs(u, v, theta, delta);
    let terms = [
      (1.0 / theta - 1.0) * ln_1mt,
      -(1.0 / delta + 1.0) * ln_s,
      -(delta + 1.0) * ln_b,
      (theta - 1.0) * (-v).ln_1p(),
    ];
    (terms, ln_s, ln_t)
  }

  /// `ln ∂_v C` on the open square: the expanded terms' sum while `s − 1`, `−ln T` are normal and the terms stay below
  /// `EXPANDED_LIMIT`, `ln_h_log` past it.
  fn ln_h_interior(u: f64, v: f64, theta: f64, delta: f64) -> f64 {
    let (terms, ln_s, ln_t) = Self::expanded(u, v, theta, delta);
    let normal = ln_s >= f64::MIN_POSITIVE && -ln_t >= f64::MIN_POSITIVE;
    if normal && terms.iter().all(|t| t.abs() <= EXPANDED_LIMIT) {
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
      if theta == 1.0 {
        (delta + 1.0) * u.ln()
      } else {
        f64::NEG_INFINITY
      }
    } else {
      Self::ln_h_log(u, v, theta, delta)
    }
  }

  /// `ln h = −((θ − 1)/θ)R − ((δ + 1)/δ) ln(1 + Q)`, `Q = F(α)/(1 + F(β))`, `F(x) = expm1(δw)`, `w = −ln(1 − x)`,
  /// `α, β = ū^θ, v̄^θ`, and `R = ln((1 − T)/β) = softplus(ln((b/β)(1 − (1 + Q)^{−1/δ})))` from `T = b(1 + Q)^{−1/δ}`.
  fn ln_h_log(u: f64, v: f64, theta: f64, delta: f64) -> f64 {
    let ln_delta = delta.ln();
    let (ln_u_bar, ln_v_bar) = ((-u).ln_1p(), (-v).ln_1p());
    let (ln_alpha, ln_beta) = (theta * ln_u_bar, theta * ln_v_bar);
    let (w_alpha, ln_w_alpha) = Self::w_of(theta, ln_u_bar, ln_alpha);
    let w_beta = -ln_one_minus_pow(theta, ln_v_bar);
    let (z_alpha, z_beta) = (delta * w_alpha, delta * w_beta);
    let ln_z_alpha = ln_delta + ln_w_alpha;
    let w_gap = ln_one_minus_pow_ratio(theta, ln_v_bar, ln_u_bar);
    // `ln Q = δ(w_α − w_β) + ln(1 − e^{−δw_α})`; with `Q < 1` and a small `δw_α`, the `1/δ` multiplies `Q`, not its log.
    let ln_q = delta * w_gap + ln_one_minus_exp_from(z_alpha, ln_z_alpha);
    let second = if ln_q < 0.0 && z_alpha < 1.0 {
      let q = ln_q.exp();
      (delta + 1.0) * w_alpha * expm1_over(z_alpha) * (-z_beta).exp() * ln_1p_over(q)
    } else {
      (delta + 1.0) / delta * softplus(ln_q)
    };
    if second == f64::INFINITY {
      return f64::NEG_INFINITY;
    }
    // `ln(w_α/β)`, from `θ ln(ū/v̄)` where both powers are below `2⁻⁵³`, so the step at `u = v` survives underflow.
    let lambda = if ln_alpha < -36.7 && ln_beta < -36.7 {
      theta * ln_ratio_complement(u, v) + 0.5 * ln_alpha.exp()
    } else {
      ln_w_alpha - ln_beta
    };
    let (ln_l_over_q, ln_l) = if ln_q < 0.0 {
      let over = ln_1p_over(ln_q.exp()).ln();
      (over, ln_q + over)
    } else {
      let ln_l = softplus(ln_q).ln();
      (ln_l - ln_q, ln_l)
    };
    let ln_z = ln_l - ln_delta;
    // `ln(w_α/β) + δ(w_α − w_β) + ln φ(δw_α) + ln(L/Q) + ln φ(L/δ) − w_β`, `L = ln(1 + Q)`: past the first, each term
    // is `≤ 0` where `α < β`, so a small `R`, which `1 − h` and so the upper quantiles hang on, keeps its digits.
    let x =
      lambda + delta * w_gap + ln_phi(z_alpha, ln_z_alpha) + ln_l_over_q + ln_phi(ln_z.exp(), ln_z)
        - w_beta;
    -(theta - 1.0) / theta * softplus(x) - second
  }

  /// `(w, ln w)` for `w = −ln(1 − x)`, `x = e^{ln x}`: `ln w = ln x + x/2` to rounding below `x = 2⁻⁵³`, where `w` may
  /// underflow with `x` while its logarithm stays exact.
  fn w_of(theta: f64, ln_x_bar: f64, ln_x: f64) -> (f64, f64) {
    let w = -ln_one_minus_pow(theta, ln_x_bar);
    let ln_w = if ln_x < -36.7 {
      ln_x + 0.5 * ln_x.exp()
    } else {
      w.ln()
    };
    (w, ln_w)
  }

  fn edge_log_density(u: f64, v: f64, theta: f64, delta: f64) -> f64 {
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
    let delta = self.delta;
    // For `1 ≤ θ ≤ 60` and `1e-147 ≤ δ ≤ 100` every node keeps `ln s` normal and each exponential below `e^617`, so the
    // plain kernel needs no check there; `h_scalar` carries `δ` outside.
    let kendall_tau = |theta: f64| {
      if (1e-147..=100.0).contains(&delta) {
        kendall_tau_numeric(
          |u, v| {
            let (t, ..) = Self::expanded(u, v, theta, delta);
            (t[0] + t[1] + t[2] + t[3]).exp()
          },
          100,
        )
      } else {
        kendall_tau_numeric(|u, v| Self::h_scalar(u, v, theta, delta), 100)
      }
    };
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
    let start_theta = self.mle_start(tau.max(0.05)).clamp(1.0, 20.0);
    let (theta, delta) = fit_two_parameters(
      u,
      v,
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

  /// `φ(t) = (1 − (1 − t)^θ)^{−δ} − 1` as `expm1(−δ ln(1 − (1 − t)^θ))` from `ln_1p(−t)`, exact where `δ` is small or
  /// `(1 − t)^θ` underflows.
  fn generator(&self, t: ArrayView1<f64>) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    Ok(t.mapv(|t| (-delta * ln_one_minus_pow(theta, (-t).ln_1p())).exp_m1()))
  }

  fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    Ok(copula_pdf(
      x,
      |u, v| Self::log_density_scalar(u, v, theta, delta),
      |u, v| Self::edge_log_density(clip(u), clip(v), theta, delta).exp(),
    ))
  }

  fn log_pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    Ok(copula_log_pdf(
      x,
      |u, v| Self::log_density_scalar(u, v, theta, delta),
      |u, v| Self::edge_log_density(clip(u), clip(v), theta, delta).exp(),
    ))
  }

  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    Ok(copula_cdf(x, |u, v| Self::cdf_scalar(u, v, theta, delta)))
  }

  /// `∂_v C(u, v)`, NaN for `v` outside `[0, 1]`.
  fn partial_derivative(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    Ok(conditional_cdf(x, |u, v| {
      Self::h_scalar(u, v, theta, delta)
    }))
  }

  /// `numerical_quantiles` of `Bb7::ln_h`; NaN for `y` or `v` outside `[0, 1]`.
  fn percent_point(
    &self,
    y: ArrayView1<f64>,
    conditioning: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    let (theta, delta) = self.params()?;
    numerical_quantiles(y, conditioning, self.family(), |u, v| {
      Self::ln_h(u, v, theta, delta)
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
  fn fit(&mut self, x: ArrayView2<f64>) -> Result<(), CopulaError> {
    check_pairs(x)?;
    self.fit_parameters(x.column(0), x.column(1))
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

mod density;

#[cfg(test)]
mod tests;
