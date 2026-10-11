//! # Ali-Mikhail-Haq (AMH) copula
//!
//! $$
//! C_\theta(u,v) = \frac{uv}{1 - \theta (1-u)(1-v)},
//! \qquad \theta \in [-1, 1).
//! $$
//!
//! Archimedean family with generator
//! $\varphi(t) = \ln\!\left(\frac{1 - \theta(1-t)}{t}\right)$. AMH has **no**
//! tail dependence in either tail. Kendall's tau is constrained to the
//! interval $\tau \in [(5 - 8\ln 2)/3,\; 1/3] \approx [-0.1817,\; 0.3333]$ —
//! AMH is therefore appropriate when the empirical $\tau$ falls inside this
//! window. Reduces to the independence copula at $\theta = 0$.
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
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::invert_tau;
use crate::traits::bivariate::tau_out_of_range;

#[derive(Debug, Clone)]
pub struct Amh {
  pub r#type: BivariateFamily,
  pub theta: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Default for Amh {
  fn default() -> Self {
    Self {
      r#type: BivariateFamily::Amh,
      theta: None,
      theta_bounds: (-1.0, 1.0),
      invalid_thetas: vec![],
    }
  }
}

impl Amh {
  pub fn new() -> Self {
    Self::default()
  }

  /// The root in `[0, 1]` of `Au² + Bu + C = 0` (`y D² = u(1 − θ + θu)`, `p = θ(1 − v)`) chosen by `B`'s sign, for
  /// `θ ≠ 1`; `√(B² − 4AC)` is `hypot((1 − θ)√(1 − y), (1 − θ + 2θv)√y)`, free of cancellation and underflow.
  pub(crate) fn inverse(theta: f64) -> impl Fn(f64, f64) -> f64 {
    move |y, v| {
      let (minus_a, b, c, z) = Self::quadratic(theta, y, v);
      let x = (1.0 - theta) * (1.0 - y).sqrt();
      // `hypot` without the call: libm rounds it this way when no square needs scaling, as `big ≥ 1e-24` for `θ < 1`.
      let (big, small) = if x >= z { (x, z) } else { (z, x) };
      let root = small.mul_add(small, big * big).sqrt();
      let (numerator, denominator) = if b < 0.0 {
        (2.0 * c, root - b)
      } else {
        (b + root, 2.0 * minus_a)
      };
      numerator / denominator
    }
  }

  /// `Amh::inverse` at `θ = 1`, where `hypot`'s first operand is 0, so the root is `2v√y`.
  pub(crate) fn inverse_at_one() -> impl Fn(f64, f64) -> f64 {
    move |y, v| {
      let (minus_a, b, c, root) = Self::quadratic(1.0, y, v);
      if b < 0.0 {
        2.0 * c / (root - b)
      } else if minus_a > 0.0 {
        (b + root) / (2.0 * minus_a)
      } else {
        // `A = 0` with `B ≥ 0` only at `(y, v) = (1, 0)`, where `h(· | 0)` jumps from 0 to 1 at `u = 0`.
        0.0
      }
    }
  }

  /// `(−A, B, C)` from same-signed terms, `1 − p = (1 − θ) + θv`, and `hypot`'s second operand `(1 − θ + 2θv)√y`.
  fn quadratic(theta: f64, y: f64, v: f64) -> (f64, f64, f64, f64) {
    let p = theta * (1.0 - v);
    let one_minus_p = (1.0 - theta) + theta * v;
    let minus_a = theta * ((1.0 - y) + y * ((1.0 - theta) + theta * v * (2.0 - v)));
    let b = 2.0 * y * p * one_minus_p - (1.0 - theta);
    let c = y * one_minus_p * one_minus_p;
    (minus_a, b, c, (1.0 - theta + 2.0 * theta * v) * y.sqrt())
  }
}

impl BivariateExt for Amh {
  fn family(&self) -> BivariateFamily {
    self.r#type
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

  /// Archimedean generator $\varphi(t) = \ln\frac{1 - \theta(1-t)}{t}$.
  fn generator(&self, t: ArrayView1<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
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

  /// Density $c(u,v) = \dfrac{(1-\theta) D + 2\theta u v}{D^3}$ where
  /// $D = 1 - \theta(1-u)(1-v)$.
  fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let u_col = x.column(0);
    let v_col = x.column(1);
    let theta = self.theta.unwrap();
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i];
      if u <= 0.0 || u >= 1.0 || v <= 0.0 || v >= 1.0 {
        out[i] = 0.0;
        continue;
      }
      let d = 1.0 - theta * (1.0 - u) * (1.0 - v);
      let num = (1.0 - theta) * d + 2.0 * theta * u * v;
      out[i] = num / (d * d * d);
    }
    Ok(out)
  }

  /// CDF $C(u,v) = uv / [1 - \theta(1-u)(1-v)]$.
  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let u_col = x.column(0);
    let v_col = x.column(1);
    let theta = self.theta.unwrap();
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i];
      if u <= 0.0 || v <= 0.0 {
        out[i] = 0.0;
        continue;
      }
      if u >= 1.0 {
        out[i] = v;
        continue;
      }
      if v >= 1.0 {
        out[i] = u;
        continue;
      }
      out[i] = u * v / (1.0 - theta * (1.0 - u) * (1.0 - v));
    }
    Ok(out)
  }

  /// $\partial_v C(u,v) = u\big(1 - \theta(1-u)\big) / D^2$, $D = 1 - \theta(1-u)(1-v)$, the derivative in the
  /// conditioning second argument; NaN for `v` outside `[0, 1]`.
  fn partial_derivative(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    Ok(conditional_cdf(x, |u, v| {
      if u <= 0.0 {
        return 0.0;
      }
      if u >= 1.0 {
        return 1.0;
      }
      let d = 1.0 - theta * (1.0 - u) * (1.0 - v);
      u * (1.0 - theta * (1.0 - u)) / (d * d)
    }))
  }

  /// `Amh::inverse`'s root, `Amh::inverse_at_one`'s at `θ = 1`; NaN for `y` or `v` outside `[0, 1]`.
  fn percent_point(
    &self,
    y: ArrayView1<f64>,
    v: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    if theta == 1.0 {
      conditional_quantiles(y, v, Self::inverse_at_one())
    } else {
      conditional_quantiles(y, v, Self::inverse(theta))
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
    if tau == lo {
      return Ok(-1.0);
    }
    if tau == hi {
      return Ok(1.0);
    }
    if tau.abs() < 1e-12 {
      return Ok(0.0);
    }
    invert_tau(|theta| self.tau_from_theta(theta) - tau, -1.0, 1.0, 1e-14)
  }

  /// AMH has no tail dependence in either tail, for any $\theta \in
  /// [-1, 1)$.
  /// Reference: Nelsen, R.B. (2006), Example 4.23 / Table 4.1 (module
  /// header).
  fn tail_dependence(&self) -> TailDependence<f64> {
    self.assert_theta_valid_for_tail_dependence();
    TailDependence {
      lower: 0.0,
      upper: 0.0,
    }
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
    let mut c = Amh::new();
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
    let mut c = Amh::new();
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
    let mut c = Amh::new();
    c.set_theta(0.7);
    let x = array![[0.4_f64, 1.0], [1.0, 0.2]];
    let cdf = c.cdf(x.view()).unwrap();
    assert!(approx(cdf[0], 0.4, 1e-12));
    assert!(approx(cdf[1], 0.2, 1e-12));
  }

  /// `τ(−1) = (5 − 8 ln 2)/3` and `τ(1) = 1/3` close the family's range.
  #[test]
  fn amh_tau_from_theta_reaches_the_paper_bounds() {
    let c = Amh::new();
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
    let c = Amh::new();
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
    let c = Amh::new();
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
    let c = Amh::new();
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
    let c = Amh::new();
    let theta = c.theta_from_tau(0.0).unwrap();
    assert!(approx(theta, 0.0, 1e-6), "τ=0 should give θ=0, got {theta}");
  }

  #[test]
  fn amh_tail_dependence_is_zero() {
    let mut c = Amh::new();
    c.set_theta(0.5);
    let td = c.tail_dependence();
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
    let mut c = Amh::new();
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
    let mut c = Amh::new();
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
