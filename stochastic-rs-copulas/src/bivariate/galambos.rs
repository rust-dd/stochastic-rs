//! # Galambos (1975) extreme-value copula
//!
//! $$
//! C_\theta(u,v) = u v \cdot \exp\!\Big\{\big[(-\ln u)^{-\theta} +
//!   (-\ln v)^{-\theta}\big]^{-1/\theta}\Big\},
//! \qquad \theta \in (0, \infty).
//! $$
//!
//! Negative-logistic max-stable family with Pickands dependence function
//! $$
//! A(t) = 1 - \big(t^{-\theta} + (1-t)^{-\theta}\big)^{-1/\theta},
//! \qquad t \in [0, 1].
//! $$
//! $\theta \to 0$ recovers the independence copula ($A \equiv 1$); $\theta
//! \to \infty$ recovers the comonotone copula ($A(t) = \max(t, 1-t)$).
//!
//! Tail dependence: $\lambda_U = 2^{-1/\theta}$ (upper),
//! $\lambda_L = 0$ (lower).
//!
//! Reference: Galambos, J. (1975), "Order statistics of samples from
//! multivariate distributions", *JASA* 70(351), 674-680.
//! Reference: Joe, H. (1997), "Multivariate Models and Dependence
//! Concepts", Chapman & Hall, §5.4.
//! Reference: Gudendorf, G. & Segers, J. (2010), "Extreme-value copulas", in Copula Theory and Its Applications, Lecture Notes in Statistics, Springer, arXiv:0911.1015, §4 (Kendall's τ from the Pickands function).

use std::f64::consts::LN_2;

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;

use crate::bivariate::BivariateFamily;
use crate::bivariate::conditional::conditional_cdf;
use crate::bivariate::conditional::conditional_quantiles;
use crate::bivariate::conditional::copula_cdf;
use crate::bivariate::conditional::copula_pdf;
use crate::bivariate::conditional::ln_add_exp;
use crate::bivariate::conditional::ln_log_ratio;
use crate::bivariate::conditional::ln_one_minus_exp;
use crate::bivariate::conditional::ln_ratio;
use crate::bivariate::conditional::log_score_parts;
use crate::bivariate::conditional::numerical_quantiles;
use crate::bivariate::conditional::on_open_square;
use crate::bivariate::conditional::softplus;
use crate::bivariate::integral::ln_integral_from_peak;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::admitted;
use crate::traits::bivariate::checked_percent_point;
use crate::traits::bivariate::invert_tau;
use crate::traits::bivariate::tau_out_of_range;

/// The ends of the τ inversion's bracket in `θ`: below `1e-4`, `τ < e^{−6900}`, and past `1e17`, `1 − τ < 2⁻⁵⁴`.
const THETA_BRACKET: (f64, f64) = (1e-4, 1e17);

#[derive(Debug, Clone, Default)]
pub struct Galambos {
  theta: Option<f64>,
}

impl Galambos {
  /// A Galambos copula; panics unless `0 <= theta < ∞`.
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

  /// `ln ∂_v C = −x + T + ln(1 − (T/y)^{θ+1})`, `x, y = −ln u, −ln v`, `T = m(1 + (m/M)^θ)^{−1/θ}`, `m ≤ M` the two: through
  /// `expm1`/`ln_1p` of `(m/M)^θ`, whose logarithm `ln_log_ratio` keeps near `m = M`; the limit laws `δ₀`, `δ₁` at `v = 0, 1`.
  pub(crate) fn ln_h(theta: f64) -> impl Fn(f64, f64) -> f64 {
    let exponent = 1.0 + 1.0 / theta;
    move |u, v| {
      if u >= 1.0 || v <= 0.0 {
        return 0.0;
      }
      if u <= 0.0 || v >= 1.0 {
        return f64::NEG_INFINITY;
      }
      let (x, y) = (-u.ln(), -v.ln());
      // The branch follows the sign of the exact `ln(x/y)`: `x` and `y` can round to one value while `(x/y)^θ` is not 1.
      let q = ln_log_ratio(u, v, x, y);
      if q <= 0.0 {
        let s = (theta * q).exp();
        let l = s.ln_1p();
        x * (-l / theta).exp_m1() + (-s * q.exp() * (-exponent * l).exp()).ln_1p()
      } else {
        let s = (-theta * q).exp();
        let l = s.ln_1p();
        // `1 − (1 + s)^{−c}` is `cs` to rounding where `s` leaves the normal range, and `ln s = −θq` keeps its digits.
        let ln_tail = if s >= f64::MIN_POSITIVE {
          ln_one_minus_exp(-exponent * l)
        } else {
          exponent.ln() - theta * q
        };
        ln_ratio(u, v) + y * (-l / theta).exp_m1() + ln_tail
      }
    }
  }

  /// `ln c = T + θ ln(m/M) + ln(a₁a₂/s + (θ + 1)(1 + s)^{−2−1/θ}/M)`, `a₁ = 1 − (1 + s)^{−k}`, `a₂ = 1 − (s/(1 + s))^k`,
  /// `k = 1 + 1/θ`: the `(1 − (T/x)^{θ+1})` factor of the chain rule kept as `a₁/s`, its limit `k` once `s` is subnormal.
  fn ln_density(theta: f64, u: f64, v: f64) -> f64 {
    if theta == 0.0 {
      return 0.0;
    }
    let (small, big, ln_ratio, ell) = log_score_parts(theta, u, v);
    let (k, s, ln_s) = (
      1.0 + 1.0 / theta,
      (theta * ln_ratio).exp(),
      theta * ln_ratio,
    );
    let a1_over_s = if s >= f64::MIN_POSITIVE {
      -(-k * ell).exp_m1() / s
    } else {
      k
    };
    let a2 = -(k * (ln_s - ell)).exp_m1();
    let ln_tail = (theta + 1.0).ln() - (1.0 + k) * ell - big.ln();
    small * (-ell / theta).exp() + ln_s + ln_add_exp((a1_over_s * a2).ln(), ln_tail)
  }

  /// Pickands dependence function
  /// $A(t) = 1 - (t^{-\theta} + (1-t)^{-\theta})^{-1/\theta}$.
  pub fn pickands(theta: f64, t: f64) -> f64 {
    if t <= 0.0 || t >= 1.0 {
      return 1.0;
    }
    let s = t.powf(-theta) + (1.0 - t).powf(-theta);
    1.0 - s.powf(-1.0 / theta)
  }

  /// `(ln τ, ln(1 − τ))` from `τ = ∫ t(1 − t)A''/A dt` and, by parts, `1 − τ = ∫ (A − tA')(A + (1 − t)A')/A² dt`, over
  /// the logit `ℓ` of `t ∈ [1/2, 1]` in terms of `ρ = e^{−θℓ}`; the smaller is integrated, the other follows.
  fn ln_tau_parts(theta: f64) -> (f64, f64) {
    // `ln A`, `A = 1 − (1 − t)(1 + ρ)^{−1/θ}`, from `ln(1 − t) = −softplus(ℓ)` and `sp = ln(1 + ρ)`.
    let ln_a = |ln_one_minus_t: f64, sp: f64| ln_one_minus_exp(ln_one_minus_t - sp / theta);
    if theta <= 1.0 {
      // `τ = 2∫ (1 + θ)ρ(1 + ρ)^{−1/θ−2}(1 − t)/A dℓ`
      let ln_tau = LN_2
        + ln_integral_from_peak(
          |l: f64| {
            let w = theta * l;
            let sp = softplus(-w);
            let ln_one_minus_t = -softplus(l);
            theta.ln_1p() - w - (1.0 / theta + 2.0) * sp + ln_one_minus_t - ln_a(ln_one_minus_t, sp)
          },
          0.5,
        );
      (ln_tau, ln_one_minus_exp(ln_tau))
    } else {
      // `1 − τ = (2/θ)∫ t(1 − t)(1 − (1 + ρ)^{−k})(1 − (1 + 1/ρ)^{−k})/A² dw`, `w = θℓ`, `k = 1 + 1/θ`
      let k = 1.0 + 1.0 / theta;
      let ln_complement = LN_2 - theta.ln()
        + ln_integral_from_peak(
          |w: f64| {
            let l = w / theta;
            let sp = softplus(-w);
            let ln_one_minus_t = -softplus(l);
            let tangents = (-(-k * sp).exp_m1()).ln() + (-(-k * (w + sp)).exp_m1()).ln();
            -softplus(-l) + ln_one_minus_t + tangents - 2.0 * ln_a(ln_one_minus_t, sp)
          },
          0.5,
        );
      (ln_one_minus_exp(ln_complement), ln_complement)
    }
  }

  /// `numerical_quantiles` of `Galambos::ln_h`, `y` at `θ = 0`.
  fn unchecked_inverse(
    &self,
    y: ArrayView1<f64>,
    v: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    let theta = self.theta.unwrap();
    if theta == 0.0 {
      return Ok(conditional_quantiles(y, v, |y, _| y));
    }
    numerical_quantiles(y, v, self.family(), Self::ln_h(theta))
  }
}

impl BivariateExt for Galambos {
  fn family(&self) -> BivariateFamily {
    BivariateFamily::Galambos
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

  /// `C = uv e^T = min(u, v) e^{m expm1(−ℓ/θ)}` from `conditional::log_score_parts`, as `e^{−M} = min(u, v)`.
  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    copula_cdf(x, |u, v| {
      let (small, _, _, ell) = log_score_parts(theta, u, v);
      u.min(v) * (small * (-ell / theta).exp_m1()).exp()
    })
  }

  /// `exp` of `Galambos::ln_h`, `u` at `θ = 0`.
  fn partial_derivative(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    if theta == 0.0 {
      return conditional_cdf(x, |u, _| u);
    }
    let ln_h = Self::ln_h(theta);
    conditional_cdf(x, |u, v| ln_h(u, v).exp())
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
    v: ArrayView1<f64>,
  ) -> Option<Result<Array1<f64>, CopulaError>> {
    Some(self.unchecked_inverse(y, v))
  }

  /// Gudendorf & Segers' (2010, §4) `τ = ∫₀¹ t(1 − t)/A dA'` through `Galambos::ln_tau_parts`; 0 below `θ = 1e-4`.
  fn tau_from_theta(&self, theta: f64) -> f64 {
    if theta < THETA_BRACKET.0 {
      return 0.0;
    }
    let (ln_tau, ln_complement) = Self::ln_tau_parts(theta);
    if ln_tau < ln_complement {
      ln_tau.exp()
    } else {
      -ln_complement.exp_m1()
    }
  }

  /// Brent on `ln θ` over `THETA_BRACKET`, matching `ln τ` up to `τ = 1/2` and `ln(1 − τ)` past it, so every `τ` in
  /// `[0, 1)`, subnormal or a few ulps under 1, inverts to its relative digits.
  fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError> {
    if !(0.0..1.0).contains(&tau) {
      return Err(tau_out_of_range(tau, "0.0 <= tau < 1.0".to_string()));
    }
    let theta = if tau == 0.0 {
      0.0
    } else {
      let residual = |x: f64| {
        let (ln_tau, ln_complement) = Self::ln_tau_parts(x.exp());
        if tau <= 0.5 {
          ln_tau - tau.ln()
        } else {
          (-tau).ln_1p() - ln_complement
        }
      };
      let (lo, hi) = THETA_BRACKET;
      invert_tau(residual, lo.ln(), hi.ln(), 1e-13)?.exp()
    };
    admitted(self, tau, theta)
  }

  /// `λ_U = 2^{−1/θ} = 2(1 − A(1/2))` of [`Galambos::pickands`], `λ_L = 0` (Galambos 1975; Joe 1997, §5.4).
  fn tail_dependence(&self) -> Result<TailDependence<f64>, CopulaError> {
    self.check_theta()?;
    let theta = self.theta.unwrap();
    Ok(TailDependence {
      lower: 0.0,
      upper: 2.0_f64.powf(-1.0 / theta),
    })
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  /// `(ln τ, ln(1 − τ))`, the logarithm each inversion matches, against the τ table's second and third columns: each
  /// within `1e-12` of itself, or `f64::MIN_POSITIVE` where it is subnormal.
  #[test]
  fn the_tau_logarithms_match_the_tau_table() {
    let rows = include_str!("../../tests/fixtures/kendall_tau.txt")
      .lines()
      .filter(|l| l.starts_with("galambos "));
    for line in rows {
      let (head, values) = line.split_once(" ; ").unwrap();
      let theta: f64 = head[9..].parse().unwrap();
      let want: Vec<f64> = values
        .split(' ')
        .skip(1)
        .map(|x| x.parse().unwrap())
        .collect();
      if want[0] == f64::NEG_INFINITY {
        continue;
      }
      let (ln_tau, ln_complement) = Galambos::ln_tau_parts(theta);
      for (got, want) in [(ln_tau, want[0]), (ln_complement, want[1])] {
        let tolerance = (1e-12 * want.abs()).max(f64::MIN_POSITIVE);
        assert!(
          (got - want).abs() <= tolerance,
          "θ = {theta:e}: {got:e}, want {want:e}"
        );
      }
    }
  }

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  #[test]
  fn galambos_marginal_recovers_input() {
    let mut c = Galambos::default();
    c.set_theta(2.0);
    let x = array![[0.5_f64, 1.0], [1.0, 0.7]];
    let cdf = c.cdf(x.view()).unwrap();
    assert!(approx(cdf[0], 0.5, 1e-12));
    assert!(approx(cdf[1], 0.7, 1e-12));
  }

  #[test]
  fn galambos_zero_theta_limit_is_independence() {
    let mut c = Galambos::default();
    c.set_theta(1e-6);
    let x = array![[0.3_f64, 0.7], [0.5, 0.5]];
    let cdf = c.cdf(x.view()).unwrap();
    assert!(approx(cdf[0], 0.21, 1e-3), "got {}", cdf[0]);
    assert!(approx(cdf[1], 0.25, 1e-3), "got {}", cdf[1]);
  }

  #[test]
  fn galambos_large_theta_approaches_comonotone() {
    let mut c = Galambos::default();
    c.set_theta(50.0);
    let x = array![[0.3_f64, 0.7], [0.5, 0.5]];
    let cdf = c.cdf(x.view()).unwrap();
    // Comonotone: C(u,v) = min(u,v).
    assert!((cdf[0] - 0.3).abs() < 0.05, "got {}", cdf[0]);
    assert!((cdf[1] - 0.5).abs() < 0.05, "got {}", cdf[1]);
  }

  #[test]
  fn galambos_pickands_at_half_matches_closed_form() {
    // A(1/2) = 1 - (2 · 2^θ)^{-1/θ} = 1 - 2^{-(θ+1)/θ}.
    for &theta in &[0.5_f64, 1.0, 2.0, 5.0] {
      let a = Galambos::pickands(theta, 0.5);
      let expected = 1.0 - 2.0_f64.powf(-(theta + 1.0) / theta);
      assert!(
        approx(a, expected, 1e-12),
        "θ={theta}: A(1/2)={a}, expected {expected}"
      );
    }
  }

  #[test]
  fn galambos_upper_tail_dependence_formula() {
    // λ_U = 2 · 2^{-1/θ}. Verified via the Pickands relation
    // λ_U = 2(1 - A(1/2)).
    for &theta in &[0.5_f64, 1.0, 2.0, 5.0] {
      let lambda_u_pickands = 2.0 * (1.0 - Galambos::pickands(theta, 0.5));
      let expected = 2.0_f64.powf(-1.0 / theta);
      assert!(
        approx(lambda_u_pickands, expected, 1e-12),
        "θ={theta}: λ_U_pickands={lambda_u_pickands}, expected {expected}"
      );
    }
  }

  #[test]
  fn galambos_theta_from_tau_inverts_tau() {
    let target_theta = 1.5_f64;
    let c = Galambos::default();
    let tau = c.tau_from_theta(target_theta);
    let recovered = c.theta_from_tau(tau).unwrap();
    assert!(
      approx(recovered, target_theta, 1e-12),
      "expected {target_theta}, got {recovered}"
    );
  }

  /// `τ = 0` inverts to `θ = 0`, the independence copula `new(0.0)` builds, from `from_tau` and from a fit at `τ̂ = 0`.
  #[test]
  fn galambos_tau_zero_inverts_to_independence() {
    let c = Galambos::from_tau(0.0).unwrap();
    assert_eq!((c.theta(), c.tau()), (Some(0.0), Some(0.0)));
    let ranks = [2.0, 4.0, 1.0, 3.0];
    let tied = ndarray::Array2::from_shape_fn((4, 2), |(i, j)| {
      if j == 0 {
        (i + 1) as f64 / 5.0
      } else {
        ranks[i] / 5.0
      }
    });
    let mut fitted = Galambos::default();
    fitted.fit(tied.view()).unwrap();
    assert_eq!(fitted.theta(), Some(0.0));
    assert!(matches!(
      Galambos::from_tau(-1e-12),
      Err(CopulaError::InvalidParameter { name: "tau", .. })
    ));
  }

  #[test]
  fn galambos_pdf_positive_interior() {
    let mut c = Galambos::default();
    c.set_theta(1.5);
    let x = array![[0.25_f64, 0.75], [0.5, 0.5], [0.1, 0.9]];
    let pdf = c.pdf(x.view()).unwrap();
    for &p in pdf.iter() {
      assert!(p > 0.0 && p.is_finite(), "pdf={p}");
    }
  }

  #[test]
  fn galambos_tail_dependence_matches_pickands_formula() {
    let mut c = Galambos::default();
    c.set_theta(1.5);
    let td = c.tail_dependence().unwrap();
    let expected = 2.0 * (1.0 - Galambos::pickands(1.5, 0.5));
    assert!(approx(td.upper, expected, 1e-12), "got {}", td.upper);
    assert_eq!(td.lower, 0.0);
  }

  /// `partial_derivative` must not jump to a hardcoded constant right at
  /// the `v\to0^+` boundary: its value at `v=1e-12` should be continuous
  /// with its value at `v=1e-9` (the F8 regression this guards). Checked
  /// at `\theta=1.5`, where the pre-fix code hardcoded `0.0` even though
  /// the true limit there is close to `1` for this `\theta` (see the
  /// method's doc).
  ///
  /// Tolerance `1e-2`, not `1e-6`: measured directly against this formula
  /// at `\theta=1.5,u=0.4`, the real gap between `v=1e-12` and `v=1e-9` is
  /// `\approx 2.2e-3` (this family's boundary convergence in `v`, like
  /// Hüsler-Reiss's, is not fast enough for `1e-6` to hold across a
  /// `1000\times` shrink in `v`). `1e-2` still rejects a jump to a
  /// hardcoded constant by more than two orders of magnitude (such a jump
  /// is `\approx 0.6` here, since the true limit sits near `1`, not `0`).
  #[test]
  fn galambos_partial_derivative_no_jump_near_v_zero() {
    let mut c = Galambos::default();
    c.set_theta(1.5);
    let u = 0.4_f64;
    let a = c.partial_derivative(array![[u, 1e-12]].view()).unwrap()[0];
    let b = c.partial_derivative(array![[u, 1e-9]].view()).unwrap()[0];
    assert!(
      approx(a, b, 1e-2),
      "expected continuity near v=0: v=1e-12 -> {a}, v=1e-9 -> {b}"
    );
    assert!(
      a > 0.9,
      "theta=1.5 v->0+ limit should be close to 1, got {a}"
    );
  }

  /// `generator` has no override in this family, so this exercises
  /// `BivariateExt::generator`'s trait-default body directly.
  #[test]
  fn galambos_generator_returns_err_not_archimedean() {
    let c = Galambos::default();
    let t = array![0.5_f64, 0.8];
    assert!(c.generator(t.view()).is_err());
  }
}
