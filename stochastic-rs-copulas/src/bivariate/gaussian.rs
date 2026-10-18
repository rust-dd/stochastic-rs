//! # Bivariate Gaussian copula
//!
//! $$
//! C_\rho(u,v) = \Phi_2\big(\Phi^{-1}(u), \Phi^{-1}(v); \rho\big), \qquad \rho \in (-1, 1),
//! $$
//! where $\Phi_2(\cdot,\cdot;\rho)$ is the standard bivariate normal CDF with
//! correlation $\rho$ and $\Phi^{-1}$ the standard normal quantile.
//!
//! - **Kendall's tau:** $\tau = \tfrac{2}{\pi}\arcsin\rho$, inverted in
//!   closed form as $\rho = \sin(\pi\tau/2)$.
//!   Reference: Kruskal, W.H. (1958), "Ordinal Measures of Association",
//!   *JASA* 53(284), 814-861.
//! - **Tail dependence:** $\lambda_L = \lambda_U = 0$ for every $\rho \in
//!   (-1, 1)$ — the Gaussian copula is asymptotically independent in both
//!   tails regardless of $\rho$, in contrast to the Student-t copula
//!   ([`crate::bivariate::t_copula::TCopula`]), which has strictly positive
//!   tail dependence for every finite degrees-of-freedom parameter.
//!
//! The joint cdf is the conditional integral of `cdf.rs`, to a relative `1e-12` at every admitted $\rho$ and in both tails.
//!
//! Reference: Embrechts, P., Lindskog, F., McNeil, A.J. (2003),
//! "Modelling Dependence with Copulas and Applications to Risk
//! Management", in *Handbook of Heavy Tailed Distributions in Finance*,
//! Elsevier, ch. 8.

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::special::norm_cdf;

use crate::bivariate::BivariateFamily;
use crate::bivariate::conditional::conditional_cdf;
use crate::bivariate::conditional::conditional_quantiles;
use crate::bivariate::conditional::copula_cdf;
use crate::bivariate::conditional::copula_density;
use crate::bivariate::conditional::copula_pdf;
use crate::bivariate::conditional::normal_quantile;
use crate::bivariate::conditional::sum_minus_one;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::admitted;
use crate::traits::bivariate::tau_out_of_range;
pub(crate) use cdf::CLOSED_FORM_BUDGET;
use cdf::bivariate_normal_cdf;
use cdf::quantile_gap;

mod cdf;

/// Clamp used to evaluate [`GaussianCopula::partial_derivative`] just
/// inside `(0, 1)` at the `v` boundary instead of returning a hardcoded
/// constant there — see that method's doc for why a constant is wrong.
const BOUNDARY_EPS: f64 = 1e-12;

#[derive(Debug, Clone, Default)]
pub struct GaussianCopula {
  theta: Option<f64>,
}

impl GaussianCopula {
  /// A Gaussian copula with correlation `rho`, stored as `theta`; panics unless `-1 < rho < 1`.
  pub fn new(rho: f64) -> Self {
    Self::try_new(rho).unwrap_or_else(|e| panic!("{e}"))
  }

  /// `new` with the domain error returned instead of a panic.
  pub fn try_new(rho: f64) -> Result<Self, CopulaError> {
    let copula = Self { theta: Some(rho) };
    copula.check_theta()?;
    Ok(copula)
  }

  /// The copula whose Kendall's τ is `tau`.
  pub fn from_tau(tau: f64) -> Result<Self, CopulaError> {
    Self::try_new(Self::default().theta_from_tau(tau)?)
  }

  /// `u = Φ(Φ⁻¹(y)√(1 − ρ²) + ρΦ⁻¹(v))` before the domain rule, for `ρ ≠ 0`.
  pub(crate) fn inverse(rho: f64) -> impl Fn(f64, f64) -> f64 {
    let sqrt_one_minus_rho2 = (1.0 - rho * rho).sqrt();
    move |y, v| {
      let z = ndtri(y) * sqrt_one_minus_rho2 + rho * ndtri(v);
      // `∞ − ∞` only at corners answered by 0: level 0, or level 1 where `v ∈ {0, 1}` puts `h(· | v)`'s mass at 0.
      if z.is_nan() { 0.0 } else { norm_cdf(z) }
    }
  }
}

/// `C_ρ(u, v) = Φ₂(Φ⁻¹(u), Φ⁻¹(v); ρ)` on the open square, the bivariate cdf of both Gaussian copulas, the Fréchet bounds
/// holding the integral's rounding.
pub(crate) fn gaussian_copula_cdf(u: f64, v: f64, rho: f64) -> f64 {
  if rho == 0.0 {
    return u * v;
  }
  let (x, y) = (normal_quantile(u), normal_quantile(v));
  // The integral reads `x + y` only at `ρ < 0`, at its end `(x + y)/α`, whose error it amplifies `end²`-fold up to `38`.
  let reach = 38.0 * (2.0 * (1.0 - rho.abs())).sqrt();
  bivariate_normal_cdf(x, y, rho, || GaussianCopula::sum(u, v, x, y, reach))
    .clamp(sum_minus_one(u, v).max(0.0), u.min(v))
}

impl GaussianCopula {
  /// `x + y`, `x, y = Φ⁻¹(u), Φ⁻¹(v)`: below `max(reach, 1/max(1, |x|, |y|))` from the exact gap `u + v − 1` by
  /// `quantile_gap` (`Φ(x) − Φ(−y) = u + v − 1`); above it the plain sum loses under `εx²`.
  fn sum(u: f64, v: f64, x: f64, y: f64, reach: f64) -> f64 {
    let plain = x + y;
    if plain.abs() < reach.max(1.0 / x.abs().max(y.abs()).max(1.0)) {
      let (p, score) = if u <= v { (u, x) } else { (v, y) };
      quantile_gap(score, p, sum_minus_one(u, v), plain)
    } else {
      plain
    }
  }

  /// `x − y`, from the exact gap `u − v` below `1/max(1, |x|, |y|)`.
  fn difference(u: f64, v: f64, x: f64, y: f64) -> f64 {
    let plain = x - y;
    if plain.abs() < 1.0 / x.abs().max(y.abs()).max(1.0) {
      quantile_gap(x, u, u - v, plain)
    } else {
      plain
    }
  }

  /// `ln c = −|ρ|d²/(2(1 − ρ)(1 + ρ)) + |ρ|(x² + y²)/(2(1 + |ρ|)) − ln √((1 − ρ)(1 + ρ))`, `d = x ∓ y` with the sign of
  /// `ρ`: the quadratic form in two same-signed terms, so `ρ` near `±1` cancels only in the inputs.
  pub(crate) fn ln_density(rho: f64, u: f64, v: f64) -> f64 {
    let (x, y) = (normal_quantile(u), normal_quantile(v));
    let one_minus_rho2 = (1.0 - rho) * (1.0 + rho);
    let r = rho.abs();
    let d = if rho >= 0.0 {
      Self::difference(u, v, x, y)
    } else {
      Self::sum(u, v, x, y, 0.0)
    };
    -r * d * d / (2.0 * one_minus_rho2) + r * (x * x + y * y) / (2.0 * (1.0 + r))
      - 0.5 * one_minus_rho2.ln()
  }
}

impl BivariateExt for GaussianCopula {
  fn family(&self) -> BivariateFamily {
    BivariateFamily::Gaussian
  }

  fn theta(&self) -> Option<f64> {
    self.theta
  }

  fn set_theta(&mut self, theta: f64) {
    self.validate_theta(theta).unwrap_or_else(|e| panic!("{e}"));
    self.theta = Some(theta);
  }

  fn theta_bounds(&self) -> (f64, f64) {
    (-1.0, 1.0)
  }

  fn invalid_thetas(&self) -> Vec<f64> {
    vec![-1.0, 1.0]
  }

  fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let rho = self.theta.unwrap();
    copula_pdf(x, |u, v| Self::ln_density(rho, u, v))
  }

  fn log_pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let rho = self.theta.unwrap();
    copula_density(x, |u, v| Self::ln_density(rho, u, v))
  }

  /// `gaussian_copula_cdf` on the open square.
  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let rho = self.theta.unwrap();
    copula_cdf(x, |u, v| gaussian_copula_cdf(u, v, rho))
  }

  /// $\partial_v C(u,v) = \Phi\big((x - \rho y)/\sqrt{1-\rho^2}\big)$, $x, y = \Phi^{-1}(u), \Phi^{-1}(v)$, with `v`
  /// clamped `1e-12` inside `(0, 1)` for the edge limits, whose sign follows `ρ`.
  fn partial_derivative(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let rho = self.theta.unwrap();
    let sqrt_one_minus_rho2 = (1.0 - rho * rho).sqrt();
    conditional_cdf(x, |u, v| {
      let v = v.clamp(BOUNDARY_EPS, 1.0 - BOUNDARY_EPS);
      norm_cdf((ndtri(u) - rho * ndtri(v)) / sqrt_one_minus_rho2)
    })
  }

  /// `GaussianCopula::inverse` of [`GaussianCopula::partial_derivative`] in `u`.
  fn percent_point_unchecked(
    &self,
    y: ArrayView1<f64>,
    v: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    let rho = self.theta.unwrap();
    // The identity at `ρ = 0`, chosen once per call: a test per pair costs the sampler.
    if rho == 0.0 {
      return Ok(conditional_quantiles(y, v, |y, _| y));
    }
    Ok(conditional_quantiles(y, v, Self::inverse(rho)))
  }

  fn tau_from_theta(&self, theta: f64) -> f64 {
    2.0 / std::f64::consts::PI * theta.asin()
  }

  fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError> {
    if !(tau > -1.0 && tau < 1.0) {
      return Err(tau_out_of_range(tau, "-1.0 < tau < 1.0".into()));
    }
    // `ρ` rounds onto the excluded `±1` once `|τ| > 1 − 6e-9`.
    admitted(self, tau, (std::f64::consts::FRAC_PI_2 * tau).sin())
  }

  /// Asymptotic independence: $\lambda_L = \lambda_U = 0$ for every $\rho
  /// \in (-1, 1)$ — the classical counterexample to "correlation implies
  /// tail dependence".
  /// Reference: Embrechts, Lindskog, McNeil (2003), §3.2 (module header).
  fn tail_dependence(&self) -> Result<TailDependence<f64>, CopulaError> {
    self.check_theta()?;
    Ok(TailDependence {
      lower: 0.0,
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

  /// The countermonotone window off the grid, `ρ` within `1e-4` of `−1`, a level near 0 against one near 1: within `1e-12`
  /// of a fixed 96- and 192-node composite Gauss–Legendre rule at 60 digits, agreeing to `2e-58`.
  #[test]
  fn the_countermonotone_window_holds_off_the_grid() {
    let cells = [
      (
        -0.9999,
        9.86587645037698e-10,
        0.999999996684254,
        1.1381334977018348e-56,
      ),
      (
        -0.99995,
        6.220960574271784e-16,
        0.9999999999999982,
        2.9449919914769766e-56,
      ),
      (
        -0.99995,
        9.86587645037698e-10,
        0.9999999978210243,
        4.163594543410096e-50,
      ),
      (
        -0.99995,
        9.86587645037698e-10,
        0.999999996684254,
        1.508533801399695e-100,
      ),
      (
        -0.99999,
        6.220960574271784e-16,
        0.9999999999999986,
        2.0775677337256595e-138,
      ),
      (
        -0.99999,
        6.220960574271784e-16,
        0.9999999999999982,
        1.3595017696382188e-204,
      ),
      (
        -0.99999,
        3.1908916729108963e-14,
        0.9999999999999319,
        4.7426203454676765e-127,
      ),
      (
        -0.99999,
        3.1908916729108963e-14,
        0.9999999999999147,
        4.433830796164244e-202,
      ),
      (
        -0.99999,
        9.86587645037698e-10,
        0.9999999981824922,
        7.754875054488471e-123,
      ),
      (
        -0.99999,
        9.86587645037698e-10,
        0.9999999978210243,
        6.1045028744763634e-198,
      ),
      (
        -0.999995,
        6.220960574271784e-16,
        0.9999999999999986,
        6.1259167977910865e-257,
      ),
      (
        -0.999995,
        3.1908916729108963e-14,
        0.9999999999999319,
        6.350698233817352e-236,
      ),
      (
        -0.999995,
        9.86587645037698e-10,
        0.9999999981824922,
        7.340533050083157e-232,
      ),
    ];
    for (rho, u, v, want) in cells {
      let got = gaussian_copula_cdf(u, v, rho);
      assert!(
        (got - want).abs() <= 1e-12 * want,
        "C_{rho}({u:e}, {v}) = {got:e}, want {want:e}"
      );
    }
  }

  #[test]
  fn gaussian_marginal_recovers_input() {
    let mut c = GaussianCopula::default();
    c.set_theta(0.4);
    let x = array![[0.5_f64, 1.0], [1.0, 0.7]];
    let cdf = c.cdf(x.view()).unwrap();
    assert!(approx(cdf[0], 0.5, 1e-9));
    assert!(approx(cdf[1], 0.7, 1e-9));
  }

  /// Φ₂(0,0;ρ) = 1/4 + asin(ρ)/(2π); ρ=0.5 → 1/3 exactly.
  ///
  /// At this exactly symmetric point the two `norm_cdf` calls cancel to
  /// `0.0` in `Φ(x) + Φ(y) - 1`, so the result is the formula's own.
  #[test]
  fn gaussian_cdf_matches_closed_form_at_origin() {
    let mut c = GaussianCopula::default();
    c.set_theta(0.5);
    let x = array![[0.5_f64, 0.5]];
    let cdf = c.cdf(x.view()).unwrap()[0];
    let expected = 0.25 + (0.5_f64).asin() / (2.0 * std::f64::consts::PI);
    assert!(
      approx(cdf, expected, 1e-12),
      "got {cdf}, expected {expected}"
    );
    assert!(approx(cdf, 1.0 / 3.0, 1e-12), "got {cdf}, expected 1/3");
  }

  /// ρ = 0.6 → τ → `theta_from_tau` → ρ.
  #[test]
  fn gaussian_tau_roundtrip() {
    let rho = 0.6_f64;
    let tau = (2.0 / std::f64::consts::PI) * rho.asin();
    let c = GaussianCopula::default();
    let recovered = c.theta_from_tau(tau).unwrap();
    assert!(
      approx(recovered, rho, 1e-12),
      "got {recovered}, expected {rho}"
    );
  }

  #[test]
  fn gaussian_fit_recovers_rho_from_samples() {
    let rho_true = 0.6_f64;
    let n = 20_000usize;
    let tol = 0.02_f64;
    let mut best_err = f64::INFINITY;
    for &seed in &[2718_u64, 999, 42] {
      let mut source = GaussianCopula::default();
      source.set_theta(rho_true);
      let samples = source.sample(n, &Deterministic::new(seed)).unwrap();

      let mut fitted = GaussianCopula::default();
      fitted.fit(samples.view()).unwrap();
      let rho_hat = fitted.theta().unwrap();
      best_err = best_err.min((rho_hat - rho_true).abs());
    }
    assert!(
      best_err < tol,
      "best-of-3 |ρ̂ - ρ| = {best_err}, expected < {tol}"
    );
  }

  /// percent_point(partial_derivative(u,v), v) == u.
  ///
  /// The round trip crosses `norm_cdf` twice (once in
  /// `partial_derivative`, once in `percent_point`).
  #[test]
  fn gaussian_h_inverse_roundtrip() {
    let mut c = GaussianCopula::default();
    c.set_theta(0.35);
    let u = 0.3_f64;
    let v = 0.65_f64;
    let p = c.partial_derivative(array![[u, v]].view()).unwrap()[0];
    let u_back = c
      .percent_point(
        Array1::from_vec(vec![p]).view(),
        Array1::from_vec(vec![v]).view(),
      )
      .unwrap()[0];
    assert!(approx(u_back, u, 1e-12), "got {u_back}, expected {u}");
  }

  #[test]
  fn gaussian_tail_dependence_is_zero() {
    let mut c = GaussianCopula::default();
    c.set_theta(0.9);
    let td = c.tail_dependence().unwrap();
    assert_eq!(td.lower, 0.0);
    assert_eq!(td.upper, 0.0);
  }

  #[test]
  fn gaussian_pdf_positive_on_unit_square_interior() {
    let mut c = GaussianCopula::default();
    c.set_theta(0.3);
    let x = array![[0.25_f64, 0.75], [0.5, 0.5], [0.1, 0.9]];
    let pdf = c.pdf(x.view()).unwrap();
    for &p in pdf.iter() {
      assert!(p > 0.0 && p.is_finite(), "pdf={p}");
    }
  }

  #[test]
  fn gaussian_generator_returns_err_not_archimedean() {
    let c = GaussianCopula::default();
    let t = array![0.5_f64, 0.8];
    assert!(c.generator(t.view()).is_err());
  }

  /// `partial_derivative` must not jump to a hardcoded constant right at
  /// the `v` boundary: its value at `v=1e-12` should be continuous with
  /// its value at `v=1e-9`, tracking the true directional limit rather
  /// than snapping to a wrong-signed `0.0`/`1.0` (the F8 regression this
  /// guards). Checked at `\rho>0`, where the pre-fix code returned the
  /// limits backwards (`0.0` near `v=0`, `1.0` near `v=1`, when the true
  /// direction is the opposite — see the method's doc).
  #[test]
  fn gaussian_partial_derivative_no_jump_near_v_boundary() {
    let mut c = GaussianCopula::default();
    c.set_theta(0.7);
    let u = 0.4_f64;
    let near_zero_a = c.partial_derivative(array![[u, 1e-12]].view()).unwrap()[0];
    let near_zero_b = c.partial_derivative(array![[u, 1e-9]].view()).unwrap()[0];
    assert!(
      approx(near_zero_a, near_zero_b, 1e-6),
      "expected continuity near v=0: v=1e-12 -> {near_zero_a}, v=1e-9 -> {near_zero_b}"
    );
    let near_one_a = c
      .partial_derivative(array![[u, 1.0 - 1e-12]].view())
      .unwrap()[0];
    let near_one_b = c
      .partial_derivative(array![[u, 1.0 - 1e-9]].view())
      .unwrap()[0];
    assert!(
      approx(near_one_a, near_one_b, 1e-6),
      "expected continuity near v=1: v=1-1e-12 -> {near_one_a}, v=1-1e-9 -> {near_one_b}"
    );

    // Directional regression check: for rho > 0, the true limit is *high*
    // near v=0 and *low* near v=1 — the opposite of the pre-fix hardcoded
    // 0.0/1.0.
    assert!(
      near_zero_a > 0.99,
      "rho>0 should push the v->0+ limit toward 1, got {near_zero_a}"
    );
    assert!(
      near_one_a < 0.01,
      "rho>0 should push the v->1- limit toward 0, got {near_one_a}"
    );
  }
}
