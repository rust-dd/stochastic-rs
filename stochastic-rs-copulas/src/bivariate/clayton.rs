//! # Clayton
//!
//! $$
//! C_\theta(u,v)=\left(u^{-\theta}+v^{-\theta}-1\right)^{-1/\theta},\ \theta>0
//! $$
//!

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::Zip;

use super::BivariateFamily;
use crate::bivariate::conditional::conditional_cdf;
use crate::bivariate::conditional::conditional_quantiles;
use crate::bivariate::conditional::confine;
use crate::bivariate::conditional::copula_cdf;
use crate::bivariate::conditional::copula_pdf;
use crate::bivariate::conditional::expm1_over;
use crate::bivariate::conditional::ln_one_minus_exp;
use crate::bivariate::conditional::ln_one_minus_pow;
use crate::bivariate::conditional::ln_ratio;
use crate::bivariate::conditional::on_open_square;
use crate::bivariate::conditional::softplus;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::admitted;
use crate::traits::bivariate::checked_percent_point;
use crate::traits::bivariate::tau_out_of_range;

/// Below this `θ`, `h(u | v) = u(1 + θ ln u (1 + ln v) + O(θ²))` is `u` to rounding at every `u, v ≥ 5e-324`
/// (`θ · 744.5 · 743.5 < 2⁻⁵³`), where `θ ln u` can fall into the subnormals.
const INDEPENDENT_BELOW: f64 = 2e-22;

#[derive(Debug, Clone, Default)]
pub struct Clayton {
  theta: Option<f64>,
}

impl Clayton {
  /// A Clayton copula; panics unless `0 <= theta < ∞`.
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

  /// `∂_v C = (1 + z)^{−(1 + 1/θ)}`, `z = (v/u)^θ(1 − u^θ)`, in logarithms so neither power overflows; 1 at `v = 0`, whose
  /// conditional law is a point mass at `u = 0`.
  pub(crate) fn h(theta: f64) -> impl Fn(f64, f64) -> f64 {
    let exponent = 1.0 + 1.0 / theta;
    move |u, v| {
      if u >= 1.0 || v <= 0.0 {
        return 1.0;
      }
      if u <= 0.0 {
        return 0.0;
      }
      let ln_v_over_u = theta * ln_ratio(v, u);
      let one_minus = -(theta * u.ln()).exp_m1();
      let ln_z = ln_v_over_u + one_minus.ln();
      // `z` itself below 1, where `exp(ln z)` would turn the rounding of `ln z` into a relative error times `1/θ`.
      let ln_1p_z = if ln_z < 0.0 {
        (ln_v_over_u.exp() * one_minus).ln_1p()
      } else {
        softplus(ln_z)
      };
      (-exponent * ln_1p_z).exp()
    }
  }

  /// `δ = x + y + ln C ≥ 0`, `x, y = −ln u, −ln v`: `−ln(1 − (1 − u^θ)(1 − v^θ))/θ` from the logarithms of both factors,
  /// so a small `θ` keeps its digits and a `u^θ` below rounding its own.
  fn excess(theta: f64, u: f64, v: f64) -> f64 {
    -ln_one_minus_exp(ln_one_minus_pow(theta, u.ln()) + ln_one_minus_pow(theta, v.ln())) / theta
  }

  /// `(M, M − m, ℓ)`, `M ≥ m` the two of `x, y`, `M − m = |ln(u/v)|` to a relative rounding, and
  /// `ℓ = θ(−ln C − M) = ln(1 + e^{−θ(M − m)}(1 − e^{−θm}))`.
  fn spread(theta: f64, u: f64, v: f64) -> (f64, f64, f64) {
    let gap = ln_ratio(u, v).abs();
    let ell = (-theta * gap + ln_one_minus_pow(theta, u.max(v).ln()))
      .exp()
      .ln_1p();
    (-u.min(v).ln(), gap, ell)
  }

  /// `ln c = ln(1 + θ) + (1 + θ)(x + y) − (1 + 2θ)(−ln C)`, chosen once per `θ`: `ln(1 + θ) − θ(x + y) + (1 + 2θ)δ` up to
  /// `θ = 1`, `ln(1 + θ) − (1 + θ)(M − m) + M − (2 + 1/θ)ℓ` above, where the `θ`-sized terms would cancel.
  fn ln_density(theta: f64, u: f64, v: f64) -> f64 {
    if theta == 0.0 {
      0.0
    } else if theta <= 1.0 {
      theta.ln_1p() + theta * (u.ln() + v.ln()) + (1.0 + 2.0 * theta) * Self::excess(theta, u, v)
    } else {
      let (big, gap, ell) = Self::spread(theta, u, v);
      theta.ln_1p() - (1.0 + theta) * gap + big - (2.0 + 1.0 / theta) * ell
    }
  }

  /// `Clayton::h` at one point for any `θ > 0`, `u` below `INDEPENDENT_BELOW` (the vine pairs' inputs are never 0).
  pub(crate) fn h_at(theta: f64, u: f64, v: f64) -> f64 {
    if theta < INDEPENDENT_BELOW {
      u
    } else {
      Self::h(theta)(u, v)
    }
  }

  /// `h⁻¹(y | v)` at one point for any `θ > 0`: `Clayton::finish` of `Clayton::level`, `y` below `INDEPENDENT_BELOW`.
  pub(crate) fn inverse_at(theta: f64, y: f64, v: f64) -> f64 {
    if theta < INDEPENDENT_BELOW {
      y
    } else {
      Self::finish(theta, Self::level(theta, y), y, v, v.powf(theta))
    }
  }

  /// `a = y^{−θ/(1 + θ)} − 1` of `h⁻¹(y | v) = (1 + a/v^θ)^{−1/θ}`, as `expm1` so `y` near 1 keeps its digits.
  pub(crate) fn level(theta: f64, y: f64) -> f64 {
    (-theta / (1.0 + theta) * y.ln()).exp_m1()
  }

  /// `exp(−ln_1p(a/v^θ)/θ)`; once `v^θ` leaves the normal range or `a/v^θ` overflows, `v a^{−1/θ}` (the dropped
  /// `(1 + v^θ/a)^{−1/θ}` is 1 to rounding), `ln a = z + ln(−expm1(−z))`, `z = −θ ln y/(1 + θ)`; 0 on `v = 0`.
  pub(crate) fn finish(theta: f64, a: f64, y: f64, v: f64, v_theta: f64) -> f64 {
    let ratio = a / v_theta;
    if v_theta >= f64::MIN_POSITIVE && ratio.is_finite() {
      return (-ratio.ln_1p() / theta).exp();
    }
    if v == 0.0 {
      return 0.0;
    }
    if y == 1.0 {
      return 1.0;
    }
    let z = -theta / (1.0 + theta) * y.ln();
    v * (-(z + (-(-z).exp_m1()).ln()) / theta).exp()
  }

  /// `h⁻¹(y | v)` by `Clayton::level` and `Clayton::finish`, `y` at `θ = 0` and below `INDEPENDENT_BELOW` but at `v = 0`.
  fn unchecked_inverse(
    &self,
    y: ArrayView1<f64>,
    v_col: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    let theta = self.theta.unwrap();

    if theta == 0.0 {
      return Ok(conditional_quantiles(y, v_col, |y, _| y));
    }
    if theta < INDEPENDENT_BELOW {
      return Ok(conditional_quantiles(y, v_col, |y, v| {
        if v <= 0.0 { 0.0 } else { y }
      }));
    }

    // Whole-array passes keep each pass's transcendental calls independent, which a fused per-pair closure does not.
    let v_theta = v_col.powf(theta);
    let mut u = y.mapv(|y| Self::level(theta, y));
    Zip::from(&mut u)
      .and(y)
      .and(v_col)
      .and(&v_theta)
      .for_each(|u, &y, &v, &v_theta| *u = confine(Self::finish(theta, *u, y, v, v_theta)));
    Ok(u)
  }
}

impl BivariateExt for Clayton {
  fn family(&self) -> BivariateFamily {
    BivariateFamily::Clayton
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

  /// Nelsen's `φ(t) = (t^{−θ} − 1)/θ` as `−ln t · expm1(x)/x`, `x = −θ ln t`, exact as `θ → 0` and `−ln t` at `θ = 0`;
  /// `expm1(x)/θ` past `x = 1`, and `e^{x − ln θ}` past `ln(f64::MAX)`, where `expm1(x)` overflows before the quotient.
  fn generator(&self, t: ArrayView1<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    Ok(t.mapv(|t| {
      let minus_ln_t = -t.ln();
      let x = theta * minus_ln_t;
      if theta == 0.0 {
        minus_ln_t
      } else if x <= 1.0 {
        minus_ln_t * expm1_over(x)
      } else if x <= f64::MAX.ln() {
        x.exp_m1() / theta
      } else {
        (x - theta.ln()).exp()
      }
    }))
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

  /// `C = uv e^δ` for `θ ≤ 1` and `min(u, v) e^{−ℓ/θ}` above, the pieces of `Clayton::ln_density`; `uv` at `θ = 0`.
  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    if theta == 0.0 {
      copula_cdf(x, |u, v| u * v)
    } else if theta <= 1.0 {
      copula_cdf(x, |u, v| {
        (u.ln() + v.ln() + Self::excess(theta, u, v)).exp()
      })
    } else {
      copula_cdf(x, |u, v| {
        let (_, _, ell) = Self::spread(theta, u, v);
        u.min(v) * (-ell / theta).exp()
      })
    }
  }

  fn percent_point(
    &self,
    y: ArrayView1<f64>,
    v: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    checked_percent_point(self, y, v)
  }

  fn percent_point_unchecked(
    &self,
    y: ArrayView1<f64>,
    v_col: ArrayView1<f64>,
  ) -> Option<Result<Array1<f64>, CopulaError>> {
    Some(self.unchecked_inverse(y, v_col))
  }

  /// `Clayton::h`, `u` at `θ = 0` and below `INDEPENDENT_BELOW` but for the point mass at `v = 0`.
  fn partial_derivative(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    if theta == 0.0 {
      return conditional_cdf(x, |u, _| u);
    }
    if theta < INDEPENDENT_BELOW {
      return conditional_cdf(x, |u, v| if v <= 0.0 { 1.0 } else { u });
    }
    conditional_cdf(x, Self::h(theta))
  }

  fn tau_from_theta(&self, theta: f64) -> f64 {
    theta / (theta + 2.0)
  }

  fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError> {
    if !(0.0..1.0).contains(&tau) {
      return Err(tau_out_of_range(tau, "0.0 <= tau < 1.0".into()));
    }
    admitted(self, tau, 2.0 * tau / (1.0 - tau))
  }

  /// Lower-tail dependence $\lambda_L = 2^{-1/\theta}$; Clayton has no
  /// upper-tail dependence ($\lambda_U = 0$).
  /// Reference: Nelsen, R.B. (2006), "An Introduction to Copulas", 2nd ed.,
  /// Springer, Example 5.21 / Table 5.1.
  fn tail_dependence(&self) -> Result<TailDependence<f64>, CopulaError> {
    self.check_theta()?;
    let theta = self.theta.unwrap();
    Ok(TailDependence {
      lower: 2.0_f64.powf(-1.0 / theta),
      upper: 0.0,
    })
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  /// Clayton θ=2: λ_L = 2^{−1/2} = 0.707106781186…, λ_U = 0.
  #[test]
  fn clayton_tail_dependence_closed_form() {
    let mut c = Clayton::default();
    c.set_theta(2.0);
    let td = c.tail_dependence().unwrap();
    assert!(
      (td.lower - 2.0_f64.powf(-0.5)).abs() < 1e-12,
      "λ_L={}, expected {}",
      td.lower,
      2.0_f64.powf(-0.5)
    );
    assert_eq!(td.upper, 0.0);
  }

  /// `sample` takes `&self`: no implementation mutates during sampling, so
  /// an immutable binding must be enough to draw from it.
  #[test]
  fn bivariate_sample_takes_shared_ref() {
    let c = Clayton::default();
    let _ = c.sample(8, &Deterministic::new(7));
  }

  /// The `θ=0` regression test, mirroring
  /// `frank::tests::frank_independence_sample_kendall_tau_near_zero`
  /// exactly (same tolerance derivation — `6` standard errors of τ under
  /// the null of independence, Kendall & Gibbons (1990) — same 3-seed
  /// best-of). Before the `percent_point` fix, `θ=0` returned
  /// `V.clone()`, making `U=V` exactly for every sampled pair
  /// (comonotonic, τ≈1) instead of the independence `θ=0` is supposed to
  /// mean.
  #[test]
  fn clayton_independence_sample_kendall_tau_near_zero() {
    let n = 4000usize;
    let se = (2.0 * (2.0 * n as f64 + 5.0) / (9.0 * n as f64 * (n as f64 - 1.0))).sqrt();
    let tol = 6.0 * se;
    let c = Clayton::new(0.0);
    let best_abs_tau = [2718_u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let samples = c.sample(n, &Deterministic::new(seed)).unwrap();
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

  /// `θ=0` is Clayton's own independence limit. `pdf`/`cdf`/
  /// `partial_derivative` had no branch for it at all — mirroring the
  /// gap `frank::tests::frank_cdf_at_independence_is_uv` documents for
  /// Frank's pre-fix `cdf` — so naive substitution hit a removable
  /// `1^{-∞}` singularity instead of resolving to the independence
  /// copula. Asserted exactly (not via a loose tolerance), matching this
  /// crate's convention for independence-limit checks.
  #[test]
  fn clayton_pdf_at_independence_is_one() {
    let c = Clayton::new(0.0);
    let pdf = c.pdf(array![[0.2_f64, 0.9], [0.5, 0.5]].view()).unwrap();
    assert!(approx(pdf[0], 1.0, 1e-12), "got {}", pdf[0]);
    assert!(approx(pdf[1], 1.0, 1e-12), "got {}", pdf[1]);
  }

  #[test]
  fn clayton_cdf_at_independence_is_uv() {
    let c = Clayton::new(0.0);
    let cdf = c.cdf(array![[0.3_f64, 0.7], [0.2, 0.9]].view()).unwrap();
    assert!(approx(cdf[0], 0.21, 1e-12), "got {}", cdf[0]);
    assert!(approx(cdf[1], 0.18, 1e-12), "got {}", cdf[1]);
  }

  /// `partial_derivative` computes $\partial_v C(u,v)$ (this family's own
  /// convention, stated in [`Clayton::percent_point`]'s doc). At
  /// independence, $C(u,v)=uv \Rightarrow \partial_v C = u$ — not the
  /// pre-fix `1^{-∞}` singularity's `v^{-1}` result.
  #[test]
  fn clayton_partial_derivative_at_independence_returns_u() {
    let c = Clayton::new(0.0);
    let pd = c
      .partial_derivative(array![[0.3_f64, 0.6], [0.2, 0.9]].view())
      .unwrap();
    assert!(approx(pd[0], 0.3, 1e-12), "got {}", pd[0]);
    assert!(approx(pd[1], 0.2, 1e-12), "got {}", pd[1]);
  }

  /// `h(u | v) = v^{-θ-1} (u^{-θ} + v^{-θ} - 1)^{-(1+θ)/θ}` (Aas, Czado,
  /// Frigessi & Bakken 2009, Table 1 — the Clayton row) at θ = 2, evaluated
  /// with the same formula in Python:
  ///   th=2.0; u=0.3; v=0.6; v**(-th-1)*(u**(-th)+v**(-th)-1)**(-(1+th)/th)
  /// gives 0.10005136755229085, and 0.8004109404183268 with u and v swapped.
  /// Guards the degenerate all-infinite branch, which ndarray's inverted
  /// `is_all_infinite` used to select for every finite input.
  #[test]
  fn partial_derivative_matches_the_closed_form_h_function() {
    let mut c = Clayton::default();
    c.set_theta(2.0);
    let h = c
      .partial_derivative(array![[0.3, 0.6], [0.6, 0.3]].view())
      .unwrap();
    assert!(
      (h[0] - 0.10005136755229085).abs() < 1e-14,
      "h(0.3 | 0.6) = {}",
      h[0]
    );
    assert!(
      (h[1] - 0.8004109404183268).abs() < 1e-14,
      "h(0.6 | 0.3) = {}",
      h[1]
    );
  }
}
