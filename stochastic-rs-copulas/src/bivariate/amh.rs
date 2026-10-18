//! # Ali-Mikhail-Haq (AMH) copula
//!
//! $$
//! C_\theta(u,v) = \frac{uv}{1 - \theta (1-u)(1-v)},
//! \qquad \theta \in [-1, 1].
//! $$
//!
//! Archimedean, `φ(t) = ln((1 − θ(1 − t))/t)`, which vanishes at `θ = 1`, where AMH is Clayton(1) (`φ = (1 − t)/t`,
//! `λ_L = 1/2`; no tail dependence below it); independence at `θ = 0`; Kendall's τ in `[(5 − 8 ln 2)/3, 1/3]`.
//!
//! Reference: Ali, M.M., Mikhail, N.N., Haq, M.S. (1978), "A class of
//! bivariate distributions including the bivariate logistic",
//! *Journal of Multivariate Analysis* 8(3), 405-412.
//! Reference: Nelsen, R.B. (2006), "An Introduction to Copulas", 2nd ed.,
//! Springer, Example 4.23 / Table 4.1 (family (3)).
//! Reference: Johnson, M.E. (1987), "Multivariate Statistical Simulation", Wiley, the conditional inverse as a quadratic in `u`, DOI 10.1002/9781118150740.
//! Reference: Hofert, M., Mächler, M. & McNeil, A. J. (2011), "Likelihood inference for Archimedean copulas", arXiv:1108.6032v1, Table 2 (Kendall's τ); Journal of Multivariate Analysis 110 (2012), 133–150.

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;

use crate::bivariate::BivariateFamily;
use crate::bivariate::conditional::conditional_cdf;
use crate::bivariate::conditional::conditional_quantiles;
use crate::bivariate::conditional::copula_cdf;
use crate::bivariate::conditional::copula_density;
use crate::bivariate::conditional::copula_pdf;
use crate::bivariate::conditional::ln_add_exp;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::admitted;
use crate::traits::bivariate::invert_tau;
use crate::traits::bivariate::tau_out_of_range;

#[derive(Debug, Clone, Default)]
pub struct Amh {
  theta: Option<f64>,
}

impl Amh {
  /// An AMH copula; panics unless `-1 <= theta <= 1`.
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

  /// The root in `[0, 1]` of `Au² + Bu + C = 0` (`y D² = u(1 − θ + θu)`, `p = θ(1 − v)`) chosen by `B`'s sign, for
  /// `θ ≠ 1`; `√(B² − 4AC)` is `hypot((1 − θ)√(1 − y), (1 − θ + 2θv)√y)`, free of cancellation and underflow.
  pub(crate) fn inverse(theta: f64) -> impl Fn(f64, f64) -> f64 {
    move |y, v| Self::root::<false>(theta, y, v)
  }

  /// `Amh::inverse` for `θ > 1/2`, `y` multiplied last into `2C/(√… − B)`: `C = y(1 − p)²` trails the root by
  /// `(1 − θ)`, so near `θ = 1` it reaches the subnormals while the root is normal; at `θ ≤ 1/2` it costs at most one bit.
  pub(crate) fn inverse_near_one(theta: f64) -> impl Fn(f64, f64) -> f64 {
    move |y, v| Self::root::<true>(theta, y, v)
  }

  #[inline(always)]
  fn root<const LATE_Y: bool>(theta: f64, y: f64, v: f64) -> f64 {
    let (minus_a, b, one_minus_p, z) = Self::quadratic(theta, y, v);
    let x = (1.0 - theta) * (1.0 - y).sqrt();
    // `hypot` without the call: libm rounds it this way when no square needs scaling, as `big ≥ 1e-24` for `θ < 1`.
    let (big, small) = if x >= z { (x, z) } else { (z, x) };
    let root = small.mul_add(small, big * big).sqrt();
    if LATE_Y && b < 0.0 {
      return y * (2.0 * one_minus_p * one_minus_p / (root - b));
    }
    let (numerator, denominator) = if b < 0.0 {
      (2.0 * (y * one_minus_p * one_minus_p), root - b)
    } else {
      (b + root, 2.0 * minus_a)
    };
    numerator / denominator
  }

  /// `Amh::inverse` at `θ = 1`, where `hypot`'s first operand is 0, so the root is `2v√y`.
  pub(crate) fn inverse_at_one() -> impl Fn(f64, f64) -> f64 {
    move |y, v| {
      let (minus_a, b, one_minus_p, root) = Self::quadratic(1.0, y, v);
      if b < 0.0 {
        2.0 * (y * one_minus_p * one_minus_p) / (root - b)
      } else if minus_a > 0.0 {
        (b + root) / (2.0 * minus_a)
      } else {
        // `A = 0` with `B ≥ 0` only at `(y, v) = (1, 0)`, where `h(· | 0)` jumps from 0 to 1 at `u = 0`.
        0.0
      }
    }
  }

  /// `D = 1 − θ(1 − u)(1 − v)` from same-signed terms: `(1 − θ) + θ(u + v(1 − u))` for `θ ≥ 0`.
  fn denominator(theta: f64, u: f64, v: f64) -> f64 {
    if theta >= 0.0 {
      (1.0 - theta) + theta * (u + v * (1.0 - u))
    } else {
      1.0 - theta * (1.0 - u) * (1.0 - v)
    }
  }

  /// `ln c = ln((1 − θ)D + 2θuv) − 3 ln D` on the open square: the numerator's terms in logarithms for `θ ≥ 0`, where
  /// `uv` may underflow, and as `(1 + θ)(1 + θūv̄) − 2θ(ū + v̄)`, both terms `≥ 0`, for `θ < 0`.
  fn ln_density(theta: f64, u: f64, v: f64) -> f64 {
    let ln_d = Self::denominator(theta, u, v).ln();
    let ln_numerator = if theta >= 0.0 {
      ln_add_exp(
        (1.0 - theta).ln() + ln_d,
        (2.0 * theta).ln() + u.ln() + v.ln(),
      )
    } else {
      let (u_bar, v_bar) = (1.0 - u, 1.0 - v);
      ((1.0 + theta) * (1.0 + theta * u_bar * v_bar) - 2.0 * theta * (u_bar + v_bar)).ln()
    };
    ln_numerator - 3.0 * ln_d
  }

  /// `(−A, B, 1 − p)` from same-signed terms, `1 − p = (1 − θ) + θv` and `C = y(1 − p)²`, and `hypot`'s second operand
  /// `(1 − θ + 2θv)√y`.
  fn quadratic(theta: f64, y: f64, v: f64) -> (f64, f64, f64, f64) {
    let p = theta * (1.0 - v);
    let one_minus_p = (1.0 - theta) + theta * v;
    let minus_a = theta * ((1.0 - y) + y * ((1.0 - theta) + theta * v * (2.0 - v)));
    let b = 2.0 * y * p * one_minus_p - (1.0 - theta);
    (
      minus_a,
      b,
      one_minus_p,
      (1.0 - theta + 2.0 * theta * v) * y.sqrt(),
    )
  }
}

impl BivariateExt for Amh {
  fn family(&self) -> BivariateFamily {
    BivariateFamily::Amh
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
    vec![]
  }

  /// Archimedean generator $\varphi(t) = \ln\frac{1 - \theta(1-t)}{t}$; at `θ = 1`, where it vanishes, its limit
  /// `φ/(1 − θ) → (1 − t)/t`, Clayton(1)'s.
  fn generator(&self, t: ArrayView1<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    if theta == 1.0 {
      return Ok(t.mapv(|t| (1.0 - t) / t));
    }
    let mut out = Array1::<f64>::zeros(t.len());
    for i in 0..t.len() {
      let ti = t[i];
      if ti <= 0.0 {
        out[i] = f64::INFINITY;
        continue;
      }
      out[i] = ((1.0 - theta * (1.0 - ti)) / ti).ln();
    }
    Ok(out)
  }

  fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    copula_pdf(x, |u, v| Self::ln_density(theta, u, v))
  }

  fn log_pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    copula_density(x, |u, v| Self::ln_density(theta, u, v))
  }

  /// `C = uv/D`, `D` as in `Amh::ln_density`, as `(u/D)v`, so the product underflows only with the result.
  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    copula_cdf(x, |u, v| (u / Self::denominator(theta, u, v)) * v)
  }

  /// `∂_v C = (u/D)((1 − θ + θu)/D)`, `D = 1 − θ + θ(u + v − uv)`: same-signed terms for `θ ≥ 0` and ratios formed before
  /// the product, which underflows only with the result.
  fn partial_derivative(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    conditional_cdf(x, |u, v| {
      if u >= 1.0 {
        return 1.0;
      }
      let d = (1.0 - theta) + theta * (u + v - u * v);
      // `D = 0` only at `θ = 1`, `u = v = 0`: the atom at `u = 0` of the point mass `h(· | 0)` is there.
      if d == 0.0 {
        return 1.0;
      }
      if u <= 0.0 {
        return 0.0;
      }
      (u / d) * (((1.0 - theta) + theta * u) / d)
    })
  }

  /// `Amh::inverse`'s root, `Amh::inverse_near_one`'s above `θ = 1/2`, `Amh::inverse_at_one`'s at `θ = 1`.
  fn percent_point_unchecked(
    &self,
    y: ArrayView1<f64>,
    v: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    let theta = self.theta.unwrap();
    if theta == 1.0 {
      Ok(conditional_quantiles(y, v, Self::inverse_at_one()))
    } else if theta > 0.5 {
      Ok(conditional_quantiles(y, v, Self::inverse_near_one(theta)))
    } else {
      Ok(conditional_quantiles(y, v, Self::inverse(theta)))
    }
  }

  /// `τ = 1 − 2[θ + (1−θ)² ln(1−θ)]/(3θ²)` (Hofert, Mächler & McNeil 2011, Table 2), with its limit `1/3` at `θ = 1`
  /// and its series below `|θ| < 1e-4`.
  fn tau_from_theta(&self, theta: f64) -> f64 {
    if theta.abs() < 1e-4 {
      let t2 = theta * theta;
      return 2.0 / 9.0 * theta + t2 / 18.0 + t2 * theta / 45.0 + t2 * t2 / 90.0;
    }
    if theta == 1.0 {
      return 1.0 / 3.0;
    }
    let one_minus = 1.0 - theta;
    1.0 - 2.0 * (theta + one_minus * one_minus * one_minus.ln()) / (3.0 * theta * theta)
  }

  /// Brent on `[−1, 1]`, whose ends give the range `[(5 − 8 ln 2)/3, 1/3]` from the same `τ`, so every `τ` in it brackets.
  fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError> {
    let (lo, hi) = (self.tau_from_theta(-1.0), self.tau_from_theta(1.0));
    if !(tau >= lo && tau <= hi) {
      return Err(tau_out_of_range(tau, format!("{lo:?} <= tau <= {hi:?}")));
    }
    let theta = if tau == lo {
      -1.0
    } else if tau == hi {
      1.0
    } else if tau.abs() < 1e-12 {
      0.0
    } else {
      invert_tau(|theta| self.tau_from_theta(theta) - tau, -1.0, 1.0, 1e-14)?
    };
    admitted(self, tau, theta)
  }

  /// `λ_L = lim_{u→0} u/(1 − θ(1 − u)²)`: 0 for `θ < 1`, `1/2` at `θ = 1`, where AMH is Clayton(1); `λ_U = 0`.
  fn tail_dependence(&self) -> Result<TailDependence<f64>, CopulaError> {
    self.check_theta()?;
    let lower = if self.theta.unwrap() == 1.0 { 0.5 } else { 0.0 };
    Ok(TailDependence { lower, upper: 0.0 })
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

  #[test]
  fn amh_independence_theta_zero_reproduces_product_cdf() {
    let mut c = Amh::default();
    c.set_theta(0.0);
    let x = array![[0.3_f64, 0.7], [0.5, 0.5], [0.9, 0.1]];
    let cdf = c.cdf(x.view()).unwrap();
    for i in 0..x.nrows() {
      let expected = x[[i, 0]] * x[[i, 1]];
      assert!(approx(cdf[i], expected, 1e-12));
    }
  }

  #[test]
  fn amh_pdf_at_interior_matches_closed_form_fixture() {
    let mut c = Amh::default();
    c.set_theta(0.5);
    // Hand-computed at (u, v) = (0.5, 0.5):
    //   D = 1 - 0.5·0.5·0.5 = 0.875
    //   c = ((1-0.5)·0.875 + 2·0.5·0.5·0.5) / 0.875³
    //     = (0.4375 + 0.25) / 0.669921875
    //     = 0.6875 / 0.669921875
    //     = 1.026239…
    let x = array![[0.5_f64, 0.5]];
    let pdf = c.pdf(x.view()).unwrap();
    let expected = 0.6875_f64 / 0.875_f64.powi(3);
    assert!(
      approx(pdf[0], expected, 1e-12),
      "pdf={}, expected {expected}",
      pdf[0]
    );
  }

  #[test]
  fn amh_marginal_recovers_input() {
    let mut c = Amh::default();
    c.set_theta(0.7);
    let x = array![[0.4_f64, 1.0], [1.0, 0.2]];
    let cdf = c.cdf(x.view()).unwrap();
    assert!(approx(cdf[0], 0.4, 1e-12));
    assert!(approx(cdf[1], 0.2, 1e-12));
  }

  /// `τ(−1) = (5 − 8 ln 2)/3` and `τ(1) = 1/3` close the family's range.
  #[test]
  fn amh_tau_from_theta_reaches_the_paper_bounds() {
    let c = Amh::default();
    let tau_lo = (5.0 - 8.0 * std::f64::consts::LN_2) / 3.0;
    let at_minus_one = c.tau_from_theta(-1.0);
    assert!(
      approx(at_minus_one, tau_lo, 1e-15),
      "τ(-1) = {at_minus_one}"
    );
    assert_eq!(c.tau_from_theta(1.0), 1.0 / 3.0);
    assert!(approx(c.tau_from_theta(0.999_999), 1.0 / 3.0, 1e-3));
  }

  /// Below `|θ| = 1e-4` the series `2θ/9 + θ²/18 + θ³/45 + θ⁴/90` answers; mpmath's closed form at 50 digits.
  #[test]
  fn amh_series_matches_mpmath_below_the_switch() {
    let c = Amh::default();
    for (theta, want) in [
      (9e-5, 2.000045001620073e-5),
      (-9e-5, -1.9999550016199272e-5),
    ] {
      let tau = c.tau_from_theta(theta);
      assert!((tau - want).abs() <= 1e-20, "θ={theta}: {tau} vs {want}");
    }
  }

  #[test]
  fn amh_theta_from_tau_inverts_tau_within_bracket() {
    let target_theta = 0.6_f64;
    let c = Amh::default();
    let tau = c.tau_from_theta(target_theta);
    let recovered = c.theta_from_tau(tau).unwrap();
    assert!(
      approx(recovered, target_theta, 1e-6),
      "expected {target_theta}, got {recovered}"
    );
  }

  /// Every `τ` of the range inverts, its ends included, where a bracket short of `±1` would leave no sign change.
  #[test]
  fn amh_inverts_the_ends_of_its_range() {
    let c = Amh::default();
    let (lo, hi) = (c.tau_from_theta(-1.0), 1.0 / 3.0);
    for tau in [lo, lo + 1e-12, -1e-9, 1e-9, hi - 1e-12] {
      let theta = c.theta_from_tau(tau).unwrap();
      assert!((-1.0..=1.0).contains(&theta), "τ = {tau}: θ = {theta}");
      assert!(
        (c.tau_from_theta(theta) - tau).abs() < 1e-10,
        "τ = {tau}: θ = {theta}"
      );
    }
  }

  #[test]
  fn amh_tau_zero_recovers_independence() {
    let c = Amh::default();
    let theta = c.theta_from_tau(0.0).unwrap();
    assert!(approx(theta, 0.0, 1e-6), "τ=0 should give θ=0, got {theta}");
  }

  /// AMH(1) is Clayton(1): `C(u, u)/u = 1/(2 − u) → 1/2`, a lower tail no `θ < 1` has, and `(1 − t)/t` generates it.
  #[test]
  fn amh_at_theta_one_is_clayton_one() {
    let amh = Amh::new(1.0);
    let clayton = crate::bivariate::clayton::Clayton::new(1.0);
    let half = TailDependence {
      lower: 0.5,
      upper: 0.0,
    };
    assert_eq!(amh.tail_dependence(), Ok(half));
    assert_eq!(clayton.tail_dependence(), Ok(half));
    assert_eq!(Amh::new(1.0 - 1e-9).tail_dependence().unwrap().lower, 0.0);
    let u = 1e-6;
    let diagonal = amh.cdf(array![[u, u]].view()).unwrap()[0] / u;
    assert!((diagonal - 0.5).abs() < 1e-6, "C(u, u)/u = {diagonal}");
    let points = array![[0.2, 0.7], [0.5, 0.5], [0.9, 0.05]];
    let phi = amh.generator(points.column(0)).unwrap() + amh.generator(points.column(1)).unwrap();
    let generated = phi.mapv(|s| 1.0 / (1.0 + s));
    let closed = amh.cdf(points.view()).unwrap();
    for (g, c) in generated.iter().zip(&closed) {
      assert!((g - c).abs() < 1e-15, "φ⁻¹(φ(u) + φ(v)) = {g} vs C = {c}");
    }
  }

  #[test]
  fn amh_tail_dependence_is_zero() {
    let mut c = Amh::default();
    c.set_theta(0.5);
    let td = c.tail_dependence().unwrap();
    assert_eq!(td.lower, 0.0);
    assert_eq!(td.upper, 0.0);
  }

  /// `partial_derivative` computes $\partial_v C(u,v)$ (the same "second
  /// argument, fixed conditioning value" convention as
  /// [`crate::bivariate::frank::Frank::partial_derivative`]). At
  /// independence ($\theta=0$, $C(u,v)=uv$), $\partial_v C = u$ — not $v$.
  /// The previous formula computed $\partial_u C(u,v)$, which equals $v$
  /// at this same point — an equally plausible-looking value in
  /// isolation, which is exactly why this direct pin, not just the
  /// sampling check below, is needed.
  #[test]
  fn amh_partial_derivative_at_independence_returns_u() {
    let mut c = Amh::default();
    c.set_theta(0.0);
    let pd = c
      .partial_derivative(array![[0.3_f64, 0.6], [0.2, 0.9]].view())
      .unwrap();
    assert!(approx(pd[0], 0.3, 1e-12), "got {}", pd[0]);
    assert!(approx(pd[1], 0.2, 1e-12), "got {}", pd[1]);
  }

  /// Kendall's τ is built from the products `(u_i-u_j)(v_i-v_j)`, which is
  /// exactly symmetric under swapping every pair's two coordinates, so τ
  /// takes the identical value whether a sampler draws `(U,V)` or its
  /// transpose `(V,U)`. It cannot, even in principle, catch a sampler that
  /// differentiates the wrong argument. What does catch it: the empirical
  /// joint CDF at specific, labeled, off-diagonal points — `P(U<=a,
  /// V<=b)` for `a != b`, compared against the closed-form `C(a,b)` —
  /// because that statistic is keyed to which physical column is `U` and
  /// which is `V`, not just their pairing.
  ///
  /// Before the `partial_derivative` fix, root-finding on the wrong
  /// partial routinely failed to bracket a solution (its achievable range
  /// over `u`, for fixed `v`, does not cover the sampled quantile `p`),
  /// and `percent_point_numerical`'s `unwrap_or(f64::EPSILON)` fallback
  /// clamped roughly three quarters of `U` draws to ~0 at `θ=0.6` —
  /// pushing every `emp(a,b)` here far from the true `C(a,b)`, with
  /// errors up to 0.47 (`n=20000`). This test's tolerance is nowhere near
  /// that: 6 standard errors of a Bernoulli proportion at `n=20000`,
  /// worst-case `p=0.5`.
  #[test]
  fn amh_sample_matches_closed_form_cdf_at_off_diagonal_points() {
    let n = 20_000usize;
    let se = (0.25_f64 / n as f64).sqrt();
    let tol = 6.0 * se;
    let theta = 0.6_f64;
    let mut c = Amh::default();
    c.set_theta(theta);
    let cdf_true = |a: f64, b: f64| a * b / (1.0 - theta * (1.0 - a) * (1.0 - b));
    let points = [
      (0.3_f64, 0.7_f64),
      (0.7, 0.3),
      (0.2, 0.8),
      (0.8, 0.2),
      (0.5, 0.5),
    ];

    let best_max_abs_err = [2718_u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let samples = c.sample(n, &Deterministic::new(seed)).unwrap();
        let u = samples.column(0);
        let v = samples.column(1);
        points
          .iter()
          .map(|&(a, b)| {
            let count = u
              .iter()
              .zip(v.iter())
              .filter(|&(&ui, &vi)| ui <= a && vi <= b)
              .count();
            (cdf_true(a, b) - count as f64 / n as f64).abs()
          })
          .fold(0.0_f64, f64::max)
      })
      .fold(f64::INFINITY, f64::min);

    assert!(
      best_max_abs_err < tol,
      "best max |emp - true| across 3 seeds = {best_max_abs_err}, expected < {tol} (6 SE, SE={se}, n={n})"
    );
  }
}
