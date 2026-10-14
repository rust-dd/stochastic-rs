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
//! Kendall's τ has no closed form: `τ = 1 − 4∬ ∂_u C ∂_v C du dv` (Nelsen 2006, Thm 5.1.3) is integrated numerically.
//!
//! Reference: Plackett, R.L. (1965), "A Class of Bivariate Distributions",
//! *Journal of the American Statistical Association* 60(310), 516-522.
//! Reference: Nelsen, R.B. (2006), "An Introduction to Copulas", 2nd ed.,
//! Springer, Example 3.11; Theorem 5.1.3 (Kendall's τ from the partial derivatives).
//! Reference: Johnson, M.E. (1987), "Multivariate Statistical Simulation", Wiley, the Plackett generator (the conditional inverse), DOI 10.1002/9781118150740.

use std::sync::OnceLock;

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;

use super::two_parameter::kendall_tau_numeric;
use crate::bivariate::BivariateFamily;
use crate::bivariate::conditional::conditional_cdf;
use crate::bivariate::conditional::conditional_quantiles;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::admitted;
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

  /// `partial_derivative`'s `∂_v C(u, v)` at one point, shared with the Kendall's τ quadrature; `u` at `θ = 1`.
  fn h(u: f64, v: f64, theta: f64) -> f64 {
    let eta = theta - 1.0;
    if (eta).abs() < 1e-12 {
      return u;
    }
    let a = 1.0 + eta * (u + v);
    let d = a * a - 4.0 * u * v * theta * eta;
    let d_safe = d.max(1e-30).sqrt();
    0.5 * (1.0 - (a - 2.0 * u * theta) / d_safe)
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

  /// Density (Nelsen 2006 eq.3.3.8):
  /// $c(u,v) = \frac{\theta [1 + (\theta-1)(u + v - 2 u v)]}{\{[1 + (\theta-1)(u + v)]^2 - 4 u v \theta (\theta - 1)\}^{3/2}}$.
  fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let u_col = x.column(0);
    let v_col = x.column(1);
    let theta = self.theta.unwrap();
    let eta = theta - 1.0;
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i];
      if (eta).abs() < 1e-12 {
        out[i] = 1.0;
        continue;
      }
      let a = 1.0 + eta * (u + v);
      let d = a * a - 4.0 * u * v * theta * eta;
      if d <= 0.0 {
        out[i] = 0.0;
        continue;
      }
      let num = theta * (1.0 + eta * (u + v - 2.0 * u * v));
      out[i] = num / d.powf(1.5);
    }
    Ok(out)
  }

  /// CDF (closed form for $\theta \neq 1$).
  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let u_col = x.column(0);
    let v_col = x.column(1);
    let theta = self.theta.unwrap();
    let eta = theta - 1.0;
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i];
      if (eta).abs() < 1e-12 {
        out[i] = u * v;
        continue;
      }
      let a = 1.0 + eta * (u + v);
      let d = a * a - 4.0 * u * v * theta * eta;
      let d_safe = d.max(0.0).sqrt();
      out[i] = (a - d_safe) / (2.0 * eta);
    }
    Ok(out)
  }

  /// $\partial_v C(u,v) = \frac{1}{2}\left[1 - \frac{1 + (\theta-1)(u+v) - 2 u \theta}{\sqrt{\{\cdots\}}}\right]$, NaN
  /// for `v` outside `[0, 1]`.
  fn partial_derivative(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    Ok(conditional_cdf(x, |u, v| Self::h(u, v, theta)))
  }

  /// `Plackett::inverse`'s root; NaN for `y` or `v` outside `[0, 1]`.
  fn percent_point(
    &self,
    y: ArrayView1<f64>,
    v: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    conditional_quantiles(y, v, Self::inverse(self.theta.unwrap()))
  }

  /// `τ = 1 − 4∬ ∂_u C ∂_v C` (Nelsen 2006, Thm 5.1.3) by the 200-panel two-point Gauss–Legendre rule: `5e-9` for `θ ≤ 100`.
  fn tau_from_theta(&self, theta: f64) -> f64 {
    if (theta - 1.0).abs() < 1e-12 {
      return 0.0;
    }
    kendall_tau_numeric(|u, v| Self::h(u, v, theta), 200)
  }

  /// Brent on `ln θ`, so the bracket `[1e-6, 1e6]` keeps a relative resolution; the rule runs `9e-4` low at its ends,
  /// so a `|τ|` above `0.99666` is rejected although the family reaches `0.99753` there.
  fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError> {
    static RANGE: OnceLock<(f64, f64)> = OnceLock::new();
    let (lo, hi) = (1e-6_f64, 1e6_f64);
    let (tau_lo, tau_hi) =
      *RANGE.get_or_init(|| (self.tau_from_theta(lo), self.tau_from_theta(hi)));
    if !(tau > tau_lo && tau < tau_hi) {
      return Err(tau_out_of_range(
        tau,
        format!("{tau_lo:?} < tau < {tau_hi:?}"),
      ));
    }
    let theta = if tau.abs() < 1e-12 {
      1.0
    } else {
      let (a, b) = if tau > 0.0 {
        (0.0, hi.ln())
      } else {
        (lo.ln(), 0.0)
      };
      // `ln θ` to `1e-8`: the rule holds `τ` to `5e-9` up to `θ = 100`.
      invert_tau(|x| self.tau_from_theta(x.exp()) - tau, a, b, 1e-8)?.exp()
    };
    admitted(self, tau, theta)
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

  /// Exact values from mpmath (`1 − 4∬ ∂_u C ∂_v C`); the 200-panel rule is good to `5e-9`.
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
      assert!((tau - want).abs() < 1e-7, "θ={theta}: {tau} vs {want}");
    }
    assert_eq!(c.tau_from_theta(1.0), 0.0);
  }
}
