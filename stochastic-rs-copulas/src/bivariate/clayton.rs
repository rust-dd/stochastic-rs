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
use crate::bivariate::conditional::check_lengths;
use crate::bivariate::conditional::conditional_cdf;
use crate::bivariate::conditional::conditional_quantiles;
use crate::bivariate::conditional::confine;
use crate::bivariate::conditional::expm1_over;
use crate::bivariate::conditional::ln_ratio;
use crate::bivariate::conditional::softplus;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::admitted;
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

  /// Nelsen's `φ(t) = (t^{−θ} − 1)/θ` as `−ln t · expm1(x)/x`, `x = −θ ln t`, which keeps its digits as `θ → 0` and is the
  /// limit `−ln t` at `θ = 0`; `expm1(x)/θ` past `x = 1`.
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
      } else {
        x.exp_m1() / theta
      }
    }))
  }

  /// Density $c(u,v) = (\theta+1)(uv)^{-\theta-1}\big(u^{-\theta}+v^{-\theta}-1\big)^{-(1+2\theta)/\theta}$
  /// — matches the mixed second partial of `cdf` for every `θ` in this
  /// family's valid range, checked directly. Now special-cased at the
  /// `θ = 0` independence boundary the same way [`Clayton::percent_point`]
  /// already is: naive substitution there hits a removable singularity
  /// (`b.powf(c)` with `b = 1` and `c = -∞`, an IEEE `pow` special case
  /// that evaluates to `1.0` regardless) that happens to leave `a` — not
  /// `1.0` — as the answer, i.e. `(uv)^{-1}`.
  fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;

    let u = x.column(0);
    let v = x.column(1);

    let theta = self.theta.unwrap();

    if theta == 0.0 {
      return Ok(Array1::ones(u.len()));
    }

    let a = (theta + 1.0) * (&u * &v).powf(-theta - 1.0);
    let b = u.powf(-theta) + v.powf(-theta) - 1.0;
    let c = -(2.0 * theta + 1.0) / theta;
    Ok(a * b.powf(c))
  }

  /// At `θ=0` (independence), `C(u,v)=uv`; the general closed form
  /// `(u^{-θ}+v^{-θ}-1)^{-1/θ}` hits the same `1^{-∞}` removable
  /// singularity as `pdf` there and evaluates to the constant `1.0` for
  /// every `u,v > 0`, not `uv`.
  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;

    let u_col = x.column(0);
    let v_col = x.column(1);

    let v_all_zeros = v_col.iter().all(|&v| v == 0.0);
    let u_all_zeros = u_col.iter().all(|&u| u == 0.0);

    if v_all_zeros || u_all_zeros {
      let shape = v_col.shape();
      return Ok(Array1::zeros(shape[0]));
    }

    let theta = self.theta.unwrap();

    if theta == 0.0 {
      return Ok(&u_col * &v_col);
    }

    let mut cdfs = Array1::<f64>::zeros(u_col.len());

    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i];

      if u > 0.0 && v > 0.0 {
        cdfs[i] = (u.powf(-theta) + v.powf(-theta) - 1.0).powf(-1.0 / theta);
      } else {
        cdfs[i] = 0.0;
      }
    }

    Ok(cdfs)
  }

  /// `h⁻¹(y | v)` by `Clayton::level` and `Clayton::finish`, `y` at `θ = 0` and below `INDEPENDENT_BELOW` but at `v = 0`;
  /// NaN for `y` or `v` outside `[0, 1]`.
  fn percent_point(
    &self,
    y: ArrayView1<f64>,
    v_col: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;

    let theta = self.theta.unwrap();

    if theta == 0.0 {
      return conditional_quantiles(y, v_col, |y, _| y);
    }
    if theta < INDEPENDENT_BELOW {
      return conditional_quantiles(y, v_col, |y, v| if v <= 0.0 { 0.0 } else { y });
    }

    check_lengths(y, v_col)?;
    // Whole-array passes keep each pass's transcendental calls independent, which a fused per-pair closure does not.
    let v_theta = v_col.powf(theta);
    let mut u = y.mapv(|y| Self::level(theta, y));
    Zip::from(&mut u)
      .and(y)
      .and(v_col)
      .and(&v_theta)
      .for_each(|u, &y, &v, &v_theta| *u = confine(y, v, Self::finish(theta, *u, y, v, v_theta)));
    Ok(u)
  }

  /// `Clayton::h`, `u` at `θ = 0` and below `INDEPENDENT_BELOW` but for the point mass at `v = 0`; NaN for `v` outside
  /// `[0, 1]`.
  fn partial_derivative(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    if theta == 0.0 {
      return Ok(conditional_cdf(x, |u, _| u));
    }
    if theta < INDEPENDENT_BELOW {
      return Ok(conditional_cdf(x, |u, v| if v <= 0.0 { 1.0 } else { u }));
    }
    Ok(conditional_cdf(x, Self::h(theta)))
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
