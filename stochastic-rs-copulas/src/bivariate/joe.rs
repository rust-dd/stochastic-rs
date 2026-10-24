//! # Joe (1997) copula
//!
//! $$
//! C_\theta(u,v) = 1 - \left[(1-u)^\theta + (1-v)^\theta - (1-u)^\theta (1-v)^\theta\right]^{1/\theta},
//! \qquad \theta \in [1, \infty).
//! $$
//!
//! Archimedean upper-tail-dependent family with generator $\varphi(t) = -\ln(1 - (1-t)^\theta)$;
//! independence at $\theta = 1$.
//!
//! Reference: Joe, H. (1997), "Multivariate Models and Dependence Concepts",
//! Chapman & Hall, §5.1.
//! Reference: Nelsen, R.B. (2006), "An Introduction to Copulas", 2nd ed.,
//! Springer, Family BB1 / table 4.1.
//! Reference: Hofert, M., Mächler, M. & McNeil, A. J. (2011), "Likelihood inference for Archimedean copulas", arXiv:1108.6032v1, Table 2 (Kendall's τ); Journal of Multivariate Analysis 110 (2012), 133–150.

use std::sync::OnceLock;

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;

use crate::bivariate::BivariateFamily;
use crate::bivariate::conditional::conditional_cdf;
use crate::bivariate::conditional::conditional_quantiles;
use crate::bivariate::conditional::copula_cdf;
use crate::bivariate::conditional::copula_pdf;
use crate::bivariate::conditional::ln_add_exp;
use crate::bivariate::conditional::ln_one_minus_pow;
use crate::bivariate::conditional::ln_ratio_complement;
use crate::bivariate::conditional::numerical_quantiles;
use crate::bivariate::conditional::on_open_square;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::admitted;
use crate::traits::bivariate::checked_percent_point;
use crate::traits::bivariate::invert_tau;
use crate::traits::bivariate::tau_out_of_range;

#[derive(Debug, Clone, Default)]
pub struct Joe {
  theta: Option<f64>,
}

impl Joe {
  /// A Joe copula; panics unless `1 <= theta < ∞`.
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

  /// `ln ∂_v C = ln(1 − a) + (1/θ − 1) ln(1 − a + a/b)`, `a, b = (1 − u)^θ, (1 − v)^θ`, in logarithms of `ln_1p(−u)` and
  /// `ln_1p(−v)`, `ln(1 − a)` without forming the product `θ ln_1p(−u)` where it is subnormal; `δ₁` at `v = 1`.
  pub(crate) fn ln_h(theta: f64) -> impl Fn(f64, f64) -> f64 {
    move |u, v| {
      if u >= 1.0 {
        return 0.0;
      }
      if u <= 0.0 || v >= 1.0 {
        return f64::NEG_INFINITY;
      }
      let ln_one_minus_a = ln_one_minus_pow(theta, (-u).ln_1p());
      let ln_a_over_b = theta * ln_ratio_complement(u, v);
      ln_one_minus_a + (1.0 / theta - 1.0) * ln_add_exp(ln_one_minus_a, ln_a_over_b)
    }
  }

  /// `(t, b − t, ℓ)`, `t ≥ b` of `ln(1 − u)`, `ln(1 − v)`, the gap by `ln_ratio_complement`, `ℓ = ln S − θt` for
  /// `S = (1 − u)^θ + (1 − v)^θ − ((1 − u)(1 − v))^θ`: `ln(1 + e^{θ(b − t)}(1 − e^{θt}))`, about its larger power.
  fn parts(theta: f64, u: f64, v: f64) -> (f64, f64, f64) {
    let top = (-u.min(v)).ln_1p();
    let gap = -ln_ratio_complement(u, v).abs();
    let ell = (theta * gap + ln_one_minus_pow(theta, top)).exp().ln_1p();
    (top, gap, ell)
  }

  /// `ln S = ln(1 − (1 − (1 − u)^θ)(1 − (1 − v)^θ))`: `ln_1p` of the product while it is below `1/2`, so `S` near 1 keeps
  /// its digits, else `θt + ℓ` of `Joe::parts`, so a power below rounding keeps its own.
  fn ln_s(theta: f64, u: f64, v: f64) -> f64 {
    let ln_product = ln_one_minus_pow(theta, (-u).ln_1p()) + ln_one_minus_pow(theta, (-v).ln_1p());
    if ln_product < -std::f64::consts::LN_2 {
      (-ln_product.exp()).ln_1p()
    } else {
      let (top, _, ell) = Self::parts(theta, u, v);
      theta * top + ell
    }
  }

  /// `ln c = (θ − 1)(b − t) − t + (1/θ − 2)ℓ + ln(θ − 1 + S)`: `(θ − 1)(t + b) + (1/θ − 2) ln S` with its `θ`-sized
  /// parts cancelled exactly, so the density holds as `θ` grows and `(1 − u)^θ` underflows.
  fn ln_density(theta: f64, u: f64, v: f64) -> f64 {
    if theta == 1.0 {
      return 0.0;
    }
    let (top, gap, ell) = Self::parts(theta, u, v);
    let s = (theta * top + ell).exp();
    (theta - 1.0) * gap - top + (1.0 / theta - 2.0) * ell + (theta - 1.0 + s).ln()
  }

  /// `numerical_quantiles` of `Joe::ln_h`, `y` at `θ = 1`.
  fn unchecked_inverse(
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

  /// Archimedean generator $\varphi(t) = -\ln(1 - (1-t)^\theta)$ from `ln_1p(−t)`, so `1 − t` and `(1 − t)^θ` keep
  /// their digits at both ends of `t` and of `θ`.
  fn unchecked_generator(&self, t: ArrayView1<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    Ok(t.mapv(|t| -ln_one_minus_pow(theta, (-t).ln_1p())))
  }
}

impl BivariateExt for Joe {
  fn family(&self) -> BivariateFamily {
    BivariateFamily::Joe
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

  fn generator_unchecked(&self, t: ArrayView1<f64>) -> Option<Result<Array1<f64>, CopulaError>> {
    Some(self.unchecked_generator(t))
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

  /// `C = 1 − S^{1/θ} = −expm1(ln S/θ)`, `ln S` from `Joe::ln_s`.
  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    if theta == 1.0 {
      return copula_cdf(x, |u, v| u * v);
    }
    copula_cdf(x, |u, v| -(Self::ln_s(theta, u, v) / theta).exp_m1())
  }

  /// `exp` of `Joe::ln_h`, `u` at `θ = 1`.
  fn partial_derivative(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    if theta == 1.0 {
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
    checked_percent_point(self, y, v, Self::unchecked_inverse)
  }

  fn percent_point_unchecked(
    &self,
    y: ArrayView1<f64>,
    v: ArrayView1<f64>,
  ) -> Option<Result<Array1<f64>, CopulaError>> {
    Some(self.unchecked_inverse(y, v))
  }

  /// `τ = 1 − 4Σ_{k≥1} 1/(k(θk + 2)(θ(k − 1) + 2))` (Hofert, Mächler & McNeil 2011, Table 2): 256 terms plus the
  /// Euler–Maclaurin tail, its integral in closed form less `f(K)/2 + f'(K)/12`; 0 for `θ ≤ 1`.
  fn tau_from_theta(&self, theta: f64) -> f64 {
    if theta <= 1.0 {
      return 0.0;
    }
    // `θ²` times the k-th term, `a = 2/θ`.
    let a = 2.0 / theta;
    let term = |k: f64| 1.0 / (k * (k + a) * (k + a - 1.0));
    let head = (1..=256).map(|k| term(f64::from(k))).sum::<f64>();
    let k = 256.0;
    let z = (a - 1.0) / k;
    let ln_1p_ratio = if z == 0.0 { 1.0 } else { z.ln_1p() / z };
    let integral = ln_1p_ratio / (a * k) - (1.0 / (k + a - 1.0)).ln_1p() / a;
    let slope = -term(k) * (1.0 / k + 1.0 / (k + a) + 1.0 / (k + a - 1.0));
    let sum = head + integral - term(k) / 2.0 - slope / 12.0;
    1.0 - 4.0 * sum / (theta * theta)
  }

  fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError> {
    static CAP: OnceLock<f64> = OnceLock::new();
    let cap = *CAP.get_or_init(|| self.tau_from_theta(1e4));
    if !(tau >= 0.0 && tau < cap) {
      return Err(tau_out_of_range(tau, format!("0.0 <= tau < {cap:?}")));
    }
    let theta = if tau < 1e-12 {
      1.0
    } else {
      invert_tau(|theta| self.tau_from_theta(theta) - tau, 1.0, 1e4, 1e-14)?
    };
    admitted(self, tau, theta)
  }

  /// $\lambda_U = 2 - 2^{1/\theta}$, Gumbel's form; Joe has no lower-tail dependence.
  /// Reference: Joe, H. (1997), "Multivariate Models and Dependence Concepts", Chapman & Hall, §5.1.
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

  use super::*;

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  #[test]
  fn joe_independence_theta_one_reproduces_product_cdf() {
    let mut c = Joe::default();
    c.set_theta(1.0);
    let x = array![[0.3_f64, 0.7], [0.5, 0.5], [0.9, 0.1]];
    let cdf = c.cdf(x.view()).unwrap();
    for i in 0..x.nrows() {
      let expected = x[[i, 0]] * x[[i, 1]];
      assert!(
        approx(cdf[i], expected, 1e-12),
        "row {i}: {} vs {}",
        cdf[i],
        expected
      );
    }
  }

  #[test]
  fn joe_cdf_marginal_recovers_input() {
    let mut c = Joe::default();
    c.set_theta(3.0);
    let x = array![[0.5_f64, 1.0], [1.0, 0.4]];
    let cdf = c.cdf(x.view()).unwrap();
    assert!(approx(cdf[0], 0.5, 1e-12));
    assert!(approx(cdf[1], 0.4, 1e-12));
  }

  #[test]
  fn joe_cdf_at_corner_is_zero() {
    let mut c = Joe::default();
    c.set_theta(2.5);
    let x = array![[0.0_f64, 0.5], [0.5, 0.0]];
    let cdf = c.cdf(x.view()).unwrap();
    assert!(approx(cdf[0], 0.0, 1e-12));
    assert!(approx(cdf[1], 0.0, 1e-12));
  }

  #[test]
  fn joe_theta_from_tau_zero_tau_gives_independence() {
    let c = Joe::default();
    assert!(approx(c.theta_from_tau(0.0).unwrap(), 1.0, 1e-9));
  }

  #[test]
  fn joe_theta_from_tau_inverts_tau_via_series() {
    let c = Joe::default();
    let tau = c.tau_from_theta(2.0);
    let recovered = c.theta_from_tau(tau).unwrap();
    assert!(
      approx(recovered, 2.0, 1e-9),
      "expected 2.0, got {recovered}"
    );
  }

  /// The series of Hofert, Mächler & McNeil (2011), Table 2, summed by mpmath at 50 digits; `τ(2) = 2 − π²/6`.
  #[test]
  fn joe_tau_matches_the_cited_series() {
    let c = Joe::default();
    for (theta, want) in [
      (1.5, 0.21927246047709384),
      (2.0, 2.0 - std::f64::consts::PI * std::f64::consts::PI / 6.0),
      (3.0, 0.5179624982298888),
      (10.0, 0.8220439420773361),
    ] {
      let tau = c.tau_from_theta(theta);
      assert!((tau - want).abs() < 1e-14, "θ={theta}: {tau} vs {want}");
    }
    assert_eq!(c.tau_from_theta(1.0), 0.0);
  }

  /// `τ(1) = 0` exactly, so the bracket from `θ = 1` changes sign for every positive `τ`, the tiniest included; Brent's
  /// `1e-10` in `θ` bounds the error in `τ`.
  #[test]
  fn joe_inverts_a_tau_near_independence() {
    let c = Joe::default();
    for tau in [1e-11, 1e-8, 1e-5] {
      let theta = c.theta_from_tau(tau).unwrap();
      assert!(theta >= 1.0, "τ = {tau}: θ = {theta}");
      assert!(
        (c.tau_from_theta(theta) - tau).abs() < 1e-10,
        "τ = {tau}: θ = {theta}"
      );
    }
  }

  #[test]
  fn joe_pdf_positive_on_unit_square_interior() {
    let mut c = Joe::default();
    c.set_theta(2.0);
    let x = array![[0.25_f64, 0.75], [0.5, 0.5], [0.1, 0.9]];
    let pdf = c.pdf(x.view()).unwrap();
    for &p in pdf.iter() {
      assert!(p > 0.0 && p.is_finite(), "pdf={p}");
    }
  }

  #[test]
  fn joe_tail_dependence_matches_gumbel_form() {
    let mut c = Joe::default();
    c.set_theta(2.0);
    let td = c.tail_dependence().unwrap();
    let expected = 2.0 - 2.0_f64.powf(0.5);
    assert!(approx(td.upper, expected, 1e-12), "got {}", td.upper);
    assert_eq!(td.lower, 0.0);
  }
}
