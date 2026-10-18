//! # Plackett (1965) copula
//!
//! Defined by a **constant local odds ratio** $\theta > 0$:
//! $$
//! \frac{C(u,v)\,[1 - u - v + C(u,v)]}{[u - C(u,v)][v - C(u,v)]} = \theta.
//! $$
//! Solving the resulting quadratic gives the closed form (for $\theta \neq 1$)
//! $$
//! C(u,v) = \frac{1 + (\theta-1)(u+v) - \sqrt{[1 + (\theta-1)(u+v)]^2 - 4 u v \theta (\theta-1)}}{2(\theta-1)}.
//! $$
//! $\theta = 1$ is the independence copula; $\theta \to 0^+$ approaches the
//! lower Fréchet bound, $\theta \to \infty$ the upper Fréchet bound. No tail
//! dependence in either direction.
//!
//! Kendall's τ has no closed form: `τ = 1 − 4∬ ∂_u C ∂_v C du dv` (Nelsen 2006, Thm 5.1.3) is one integral once the
//! inner one is taken in closed form, `τ(1/θ) = −τ(θ)`.
//!
//! Reference: Plackett, R.L. (1965), "A Class of Bivariate Distributions",
//! *Journal of the American Statistical Association* 60(310), 516-522.
//! Reference: Nelsen, R.B. (2006), "An Introduction to Copulas", 2nd ed.,
//! Springer, Example 3.11; Theorem 5.1.3 (Kendall's τ from the partial derivatives).
//! Reference: Johnson, M.E. (1987), "Multivariate Statistical Simulation", Wiley, the Plackett generator (the conditional inverse), DOI 10.1002/9781118150740.

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use stochastic_rs_distributions::RealExt;

use crate::bivariate::BivariateFamily;
use crate::bivariate::conditional::conditional_cdf;
use crate::bivariate::conditional::conditional_quantiles;
use crate::bivariate::conditional::copula_cdf;
use crate::bivariate::conditional::copula_pdf;
use crate::bivariate::conditional::on_open_square;
use crate::bivariate::conditional::sum_minus_one;
use crate::bivariate::integral::rule;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::admitted;
use crate::traits::bivariate::checked_percent_point;
use crate::traits::bivariate::invert_tau;
use crate::traits::bivariate::tau_out_of_range;

#[derive(Debug, Clone, Default)]
pub struct Plackett {
  theta: Option<f64>,
}

impl Plackett {
  /// A Plackett copula; panics unless `1e-12 <= theta < ∞`.
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

  /// The root in `[0, 1]` of `bu² − cu + e = 0` (`∂_v C = y` squared, `a = y(1 − y)`), as `(c + (2y − 1)d)/(2b)` from
  /// `y = 1/2` up and the rationalised `2e/(c + (1 − 2y)d)` below, every coefficient divided by `θ` so none overflows.
  pub(crate) fn inverse(theta: f64) -> impl Fn(f64, f64) -> f64 {
    let eta = theta - 1.0;
    let eta_squared_over_theta = eta * (eta / theta);
    move |y, v| {
      let a = y * (1.0 - y);
      let c = 2.0 * a * (v * theta + (1.0 - v) / theta) + (1.0 - 2.0 * a);
      let d = (1.0 + 4.0 * a * v * (1.0 - v) * eta_squared_over_theta).sqrt();
      let e = a * ((1.0 - v) + theta * v) * ((1.0 - v) / theta + v);
      // Both roots share one division: selecting their operands first keeps either half of `y` free of a branch.
      let (numerator, denominator) = if y < 0.5 {
        (2.0 * e, c + (1.0 - 2.0 * y) * d)
      } else {
        (
          c + (2.0 * y - 1.0) * d,
          2.0 * (1.0 + a * eta_squared_over_theta),
        )
      };
      numerator / denominator
    }
  }

  /// `∂_v C = (1 − b/√d)/2`, `b = w + θ(v − u)`, `d = w² + 2θ(u(1 − u) + v(1 − v)) + θ²(u − v)²` with the exact `w = 1 − u − v`,
  /// each term over the largest so none over- or underflows; `2θu(1 − u)/(√d(√d + b))` for `b ≥ 0`, as `d − b² = 4θu(1 − u)`.
  pub(crate) fn h(theta: f64) -> impl Fn(f64, f64) -> f64 {
    move |u, v| {
      if u <= 0.0 {
        return 0.0;
      }
      if u >= 1.0 {
        return 1.0;
      }
      let w = -sum_minus_one(u, v);
      let shift = theta * (v - u);
      let spread = theta * (u * (1.0 - u) + v * (1.0 - v));
      // Unscaled wherever `d` and `θu` are normal with room to spare, the scaled form only in the corners.
      let (d, mass) = (w * w + 2.0 * spread + shift * shift, theta * u);
      if d < 1e300 && d > 1e-300 && mass > 1e-300 {
        let (root, b) = (d.sqrt(), w + shift);
        return if b >= 0.0 {
          2.0 * mass * (1.0 - u) / (root * (root + b))
        } else {
          (0.5 - 0.5 * b / root).min_or_nan(1.0)
        };
      }
      let scale = w.abs().max(shift.abs()).max(spread.sqrt());
      let (w, shift) = (w / scale, shift / scale);
      let root = (w * w + 2.0 * (spread / scale / scale) + shift * shift).sqrt();
      let b = w + shift;
      if b >= 0.0 {
        // `θu` itself is subnormal where both are small, and `u/scale` underflows where `θ` is large.
        let mass = theta * u;
        let ratio = if mass >= f64::MIN_POSITIVE {
          mass / scale / scale
        } else {
          (theta / scale) * (u / scale)
        };
        ratio * (1.0 - u) * 2.0 / (root * (root + b))
      } else {
        (0.5 - 0.5 * b / root).min_or_nan(1.0)
      }
    }
  }

  /// `Plackett::inverse`'s root.
  fn unchecked_inverse(
    &self,
    y: ArrayView1<f64>,
    v: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    Ok(conditional_quantiles(
      y,
      v,
      Self::inverse(self.theta.unwrap()),
    ))
  }
}

impl Plackett {
  /// `(w, σ, d/σ²)`, `d = a² − 4uvθ(θ − 1) = w² + 2θ(u(1 − u) + v(1 − v)) + θ²(u − v)²` with the exact `w = 1 − u − v`, `σ`
  /// the largest of `|w|`, `θ|u − v|`, `√(θ(u(1 − u) + v(1 − v)))`; `θ/σ` before each product, so none under- or overflows.
  fn scaled_d(theta: f64, u: f64, v: f64) -> (f64, f64, f64) {
    let w = -sum_minus_one(u, v);
    let r = u * (1.0 - u) + v * (1.0 - v);
    let sigma = w.abs().max(theta * (u - v).abs()).max((theta * r).sqrt());
    let (ws, ts) = (w / sigma, theta / sigma);
    let shift = ts * (u - v);
    (w, sigma, ws * ws + 2.0 * (ts * (r / sigma)) + shift * shift)
  }

  /// `C = (a − √d)/(2(θ − 1))`, `a = 1 + (θ − 1)(u + v) = w + θ(u + v)`, rationalised to `2θuv/(a + √d)` where `a ≥ 0` and
  /// scaled by the larger of `Plackett::scaled_d`'s `σ` and `θ max(u, v)`; the plain quotient where `a < 0`, so `θ < 1/2`.
  fn cdf_interior(theta: f64, u: f64, v: f64) -> f64 {
    let (w, sigma, d) = Self::scaled_d(theta, u, v);
    let scale = sigma.max(theta * u.max(v));
    let ts = theta / scale;
    let a = w / scale + ts * u + ts * v;
    let root = sigma / scale * d.sqrt();
    if a >= 0.0 {
      2.0 * (ts * u) * v / (a + root)
    } else {
      (a - root) * scale / (2.0 * (theta - 1.0))
    }
  }

  /// `ln c = ln θ + ln(p + θq) − (3/2) ln d`, `p = uv + (1 − u)(1 − v)`, `q = u(1 − v) + v(1 − u)`: `1 + (θ − 1)(u + v − 2uv)`
  /// in two terms `≥ 0` and `d` through `Plackett::scaled_d`, so neither cancels nor overflows.
  fn ln_density(theta: f64, u: f64, v: f64) -> f64 {
    let (u_bar, v_bar) = (1.0 - u, 1.0 - v);
    let (p, q) = (u * v + u_bar * v_bar, u * v_bar + u_bar * v);
    let (_, sigma, d) = Self::scaled_d(theta, u, v);
    theta.ln() + (p + theta * q).ln() - 3.0 * sigma.ln() - 1.5 * d.ln()
  }

  /// `1 − τ` at `θ ≥ 1`: `4∬ ∂_u C ∂_v C` in `s = u + v − 1`, `d = u − v`, closed in `d`, is `∫ (2L/a)(αF − θL²G) dL`,
  /// `L = 1 − s`; Gauss–Legendre in `√L`, panels halved to the corner layer `√L ≈ θ^{−1/2}`, nil past `2⁻²²`.
  fn tau_complement(theta: f64) -> f64 {
    if theta == 1.0 {
      return 1.0;
    }
    let eta = theta - 1.0;
    let integrand = |t: f64| {
      let l = t * t;
      let w = l * (2.0 - l);
      let a = 1.0 + eta * w;
      let alpha = 2.0 + (theta - 2.0) * w;
      let (f, g) = arctan_ratios(l * l * theta * (eta / a));
      4.0 * t * l / a * (alpha * f - theta * l * l * g)
    };
    let floor = (0.25 / theta.sqrt()).max(2.0_f64.powi(-22));
    let (mut hi, mut sum) = (1.0, 0.0);
    while hi > 0.0 {
      let lo = if 0.5 * hi < floor { 0.0 } else { 0.5 * hi };
      let (mid, half) = (0.5 * (hi + lo), 0.5 * (hi - lo));
      sum += half
        * rule()
          .iter()
          .map(|&(x, weight)| weight * integrand(mid + half * x))
          .sum::<f64>();
      hi = lo;
    }
    sum
  }
}

/// Below this `|ln θ|` Kendall's τ is its odd series `(2/9)s − (2/675)s³ + s⁵/66150` in `s = ln θ` (sympy, from Nelsen's
/// `1 − 4∬ C_u C_v` term by term in `θ − 1`), whose next term `s⁷/661500` stays under `1e-13` of `τ`.
const TAU_SERIES_BELOW: f64 = 0.0495;

/// `(F, G)`, `F(x) = atan(√x)/√x` and `G(x) = (1 − F(x))/x`, by `G`'s series `Σ (−x)ᵏ/(2k + 3)` below `x = 1/8`, where
/// `1 − F` would cancel.
fn arctan_ratios(x: f64) -> (f64, f64) {
  if x < 0.125 {
    let g = (0..18)
      .rev()
      .fold(0.0, |g, k| 1.0 / f64::from(2 * k + 3) - x * g);
    (1.0 - x * g, g)
  } else {
    let r = x.sqrt();
    let f = r.atan() / r;
    (f, (1.0 - f) / x)
  }
}

impl BivariateExt for Plackett {
  fn family(&self) -> BivariateFamily {
    BivariateFamily::Plackett
  }

  fn theta(&self) -> Option<f64> {
    self.theta
  }

  fn set_theta(&mut self, theta: f64) {
    self.validate_theta(theta).unwrap_or_else(|e| panic!("{e}"));
    self.theta = Some(theta);
  }

  fn theta_bounds(&self) -> (f64, f64) {
    (1e-12, f64::INFINITY)
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

  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    copula_cdf(x, |u, v| Self::cdf_interior(theta, u, v))
  }

  /// $\partial_v C(u,v) = \frac{1}{2}\left[1 - \frac{1 + (\theta-1)(u+v) - 2 u \theta}{\sqrt{\{\cdots\}}}\right]$.
  fn partial_derivative(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    conditional_cdf(x, Self::h(self.theta.unwrap()))
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
    v: ArrayView1<f64>,
  ) -> Option<Result<Array1<f64>, CopulaError>> {
    Some(self.unchecked_inverse(y, v))
  }

  /// `1 − Plackett::tau_complement(θ)`, through `τ(1/θ) = −τ(θ)` below `θ = 1`, and the series of `TAU_SERIES_BELOW` near 1.
  fn tau_from_theta(&self, theta: f64) -> f64 {
    let s = theta.ln();
    if s.abs() <= TAU_SERIES_BELOW {
      let s2 = s * s;
      return s * (2.0 / 9.0 + s2 * (-2.0 / 675.0 + s2 / 66150.0));
    }
    if theta >= 1.0 {
      1.0 - Self::tau_complement(theta)
    } else {
      Self::tau_complement(1.0 / theta) - 1.0
    }
  }

  /// Brent on `ln θ ≥ 0` for `|τ|`, matching `ln(1 − |τ|)` so a `τ` near ±1 keeps its digits, `1/θ` for `τ < 0`; the
  /// bracket ends at `θ = 1e34`, where `1 − τ` is under `2⁻⁵⁴`, and at the bound `1e-12`.
  fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError> {
    if tau == 0.0 {
      return admitted(self, tau, 1.0);
    }
    let top: f64 = if tau > 0.0 { 1e34 } else { 1e12 };
    let target = (-tau.abs()).ln_1p();
    // NaN for `|τ| > 1` or a NaN `τ`, and `−∞` at `|τ| = 1`.
    if target.is_nan() || target <= Self::tau_complement(top).ln() {
      let lowest = Self::tau_complement(1e12) - 1.0;
      return Err(tau_out_of_range(tau, format!("{lowest:?} < tau < 1.0")));
    }
    let x = invert_tau(
      |x| Self::tau_complement(x.exp()).ln() - target,
      0.0,
      top.ln(),
      1e-13,
    )?;
    admitted(self, tau, if tau > 0.0 { x.exp() } else { (-x).exp() })
  }

  /// Plackett has no tail dependence in either tail, for any $\theta > 0$.
  /// Reference: Nelsen, R.B. (2006), "An Introduction to Copulas", 2nd ed.,
  /// Springer, §2.3 / Example 3.11 (module header).
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

  use super::*;

  /// `ln(1 − |τ|)`, which the inversion matches, against the third column of the τ table: within `1e-12` of itself (or
  /// `f64::MIN_POSITIVE` below it), from `ln_1p(−|τ|)` while `|τ| < 1/2` and from the complement past it.
  #[test]
  fn the_complement_matches_the_tau_table() {
    let rows = include_str!("../../tests/fixtures/kendall_tau.txt")
      .lines()
      .filter(|l| l.starts_with("plackett "));
    for line in rows {
      let (head, values) = line.split_once(" ; ").unwrap();
      let theta: f64 = head[9..].parse().unwrap();
      let want: f64 = values.split(' ').nth(2).unwrap().parse().unwrap();
      let tau = Plackett::default().tau_from_theta(theta);
      let got = if tau.abs() < 0.5 {
        (-tau.abs()).ln_1p()
      } else {
        Plackett::tau_complement(theta.max(1.0 / theta)).ln()
      };
      let tolerance = (1e-12 * want.abs()).max(f64::MIN_POSITIVE);
      assert!(
        (got - want).abs() <= tolerance,
        "θ = {theta:e}: {got:e}, want {want:e}"
      );
    }
  }

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  #[test]
  fn plackett_independence_theta_one_reproduces_product_cdf() {
    let mut c = Plackett::default();
    c.set_theta(1.0);
    let x = array![[0.3_f64, 0.7], [0.5, 0.5]];
    let cdf = c.cdf(x.view()).unwrap();
    assert!(approx(cdf[0], 0.21, 1e-9));
    assert!(approx(cdf[1], 0.25, 1e-9));
  }

  #[test]
  fn plackett_cdf_monotone_in_theta() {
    let x = array![[0.5_f64, 0.5]];
    let mut c_low = Plackett::default();
    c_low.set_theta(0.2);
    let mut c_high = Plackett::default();
    c_high.set_theta(5.0);
    let lo = c_low.cdf(x.view()).unwrap()[0];
    let hi = c_high.cdf(x.view()).unwrap()[0];
    // At (0.5, 0.5): independence gives 0.25; θ < 1 → smaller, θ > 1 → larger.
    assert!(lo < 0.25, "θ=0.2 should give < 0.25, got {lo}");
    assert!(hi > 0.25, "θ=5.0 should give > 0.25, got {hi}");
  }

  #[test]
  fn plackett_cdf_marginal_recovers_input() {
    let mut c = Plackett::default();
    c.set_theta(3.0);
    let x = array![[0.5_f64, 1.0], [1.0, 0.4]];
    let cdf = c.cdf(x.view()).unwrap();
    assert!(approx(cdf[0], 0.5, 1e-9));
    assert!(approx(cdf[1], 0.4, 1e-9));
  }

  #[test]
  fn plackett_theta_from_tau_zero_tau_gives_independence() {
    let c = Plackett::default();
    assert!(approx(c.theta_from_tau(0.0).unwrap(), 1.0, 1e-12));
  }

  #[test]
  fn plackett_pdf_positive_on_unit_square_interior() {
    let mut c = Plackett::default();
    c.set_theta(2.0);
    let x = array![[0.25_f64, 0.75], [0.5, 0.5], [0.1, 0.9]];
    let pdf = c.pdf(x.view()).unwrap();
    for &p in pdf.iter() {
      assert!(p > 0.0 && p.is_finite(), "pdf={p}");
    }
  }

  #[test]
  fn plackett_generator_returns_err_not_archimedean() {
    let c = Plackett::default();
    let t = array![0.5_f64, 0.8];
    assert!(c.generator(t.view()).is_err());
  }

  #[test]
  fn plackett_tail_dependence_is_zero() {
    let mut c = Plackett::default();
    c.set_theta(5.0);
    let td = c.tail_dependence().unwrap();
    assert_eq!(td.lower, 0.0);
    assert_eq!(td.upper, 0.0);
  }

  /// Exact values from mpmath (`1 − 4∬ ∂_u C ∂_v C`).
  #[test]
  fn plackett_tau_matches_mpmath() {
    let c = Plackett::default();
    for (theta, want) in [
      (0.5, -0.1530484986351886),
      (2.0, 0.1530484986351886),
      (5.0, 0.3454998686389342),
      (20.0, 0.5916589746444259),
      (100.0, 0.7871967240291546),
    ] {
      let tau = c.tau_from_theta(theta);
      assert!((tau - want).abs() < 1e-15, "θ={theta}: {tau} vs {want}");
    }
    assert_eq!(c.tau_from_theta(1.0), 0.0);
  }
}
