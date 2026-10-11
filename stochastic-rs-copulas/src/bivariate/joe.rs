//! # Joe (1997) copula
//!
//! $$
//! C_\theta(u,v) = 1 - \left[(1-u)^\theta + (1-v)^\theta - (1-u)^\theta (1-v)^\theta\right]^{1/\theta},
//! \qquad \theta \in [1, \infty).
//! $$
//!
//! Archimedean upper-tail-dependent family with generator
//! $\varphi(t) = -\ln(1 - (1-t)^\theta)$. Reduces to the independence copula
//! at $\theta = 1$.
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
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::admitted;
use crate::traits::bivariate::invert_tau;
use crate::traits::bivariate::tau_out_of_range;

#[derive(Debug, Clone)]
pub struct Joe {
  pub r#type: BivariateFamily,
  pub theta: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Default for Joe {
  fn default() -> Self {
    Self {
      r#type: BivariateFamily::Joe,
      theta: None,
      theta_bounds: (1.0, f64::INFINITY),
      invalid_thetas: vec![],
    }
  }
}

impl Joe {
  pub fn new() -> Self {
    Self::default()
  }
}

impl BivariateExt for Joe {
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

  /// Archimedean generator $\varphi(t) = -\ln(1 - (1-t)^\theta)$.
  fn generator(&self, t: ArrayView1<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    let mut out = Array1::<f64>::zeros(t.len());
    for i in 0..t.len() {
      let a = (1.0 - t[i]).powf(theta);
      out[i] = -((1.0 - a).ln());
    }
    Ok(out)
  }

  /// Density. Let $a = (1-u)^\theta$, $b = (1-v)^\theta$,
  /// $S = a + b - a b$. Then
  /// $c(u,v) = (1-u)^{\theta-1}(1-v)^{\theta-1} S^{1/\theta - 2}(S + \theta - 1)$.
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
      let a = (1.0 - u).powf(theta);
      let b = (1.0 - v).powf(theta);
      let s = a + b - a * b;
      out[i] = (1.0 - u).powf(theta - 1.0)
        * (1.0 - v).powf(theta - 1.0)
        * s.powf(1.0 / theta - 2.0)
        * (s + theta - 1.0);
    }
    Ok(out)
  }

  /// CDF $C(u,v) = 1 - [(1-u)^\theta + (1-v)^\theta - (1-u)^\theta(1-v)^\theta]^{1/\theta}$.
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
      let a = (1.0 - u).powf(theta);
      let b = (1.0 - v).powf(theta);
      out[i] = 1.0 - (a + b - a * b).powf(1.0 / theta);
    }
    Ok(out)
  }

  /// `∂_v C = (1 − a)(1 − a + a/b)^{1/θ − 1}`, `a, b = (1 − u)^θ, (1 − v)^θ`, `1 − a` and `a/b` from `ln_1p(−u)` and
  /// `ln_1p(−v)`: no underflow near `u, v = 1`, no cancellation near `u = 0`; NaN for `v` outside `[0, 1]`.
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
      let log_1m_u = (-u).ln_1p();
      let one_minus_a = -(theta * log_1m_u).exp_m1();
      let a_over_b = (theta * (log_1m_u - (-v).ln_1p())).exp();
      one_minus_a * (one_minus_a + a_over_b).powf(1.0 / theta - 1.0)
    }))
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

  /// Upper-tail dependence $\lambda_U = 2 - 2^{1/\theta}$ (same functional
  /// form as Gumbel); Joe has no lower-tail dependence.
  /// Reference: Joe, H. (1997), "Multivariate Models and Dependence
  /// Concepts", Chapman & Hall, §5.1.
  fn tail_dependence(&self) -> TailDependence<f64> {
    self.assert_theta_valid_for_tail_dependence();
    let theta = self.theta.unwrap();
    TailDependence {
      lower: 0.0,
      upper: 2.0 - 2.0_f64.powf(1.0 / theta),
    }
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
    let mut c = Joe::new();
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
    let mut c = Joe::new();
    c.set_theta(3.0);
    let x = array![[0.5_f64, 1.0], [1.0, 0.4]];
    let cdf = c.cdf(x.view()).unwrap();
    assert!(approx(cdf[0], 0.5, 1e-12));
    assert!(approx(cdf[1], 0.4, 1e-12));
  }

  #[test]
  fn joe_cdf_at_corner_is_zero() {
    let mut c = Joe::new();
    c.set_theta(2.5);
    let x = array![[0.0_f64, 0.5], [0.5, 0.0]];
    let cdf = c.cdf(x.view()).unwrap();
    assert!(approx(cdf[0], 0.0, 1e-12));
    assert!(approx(cdf[1], 0.0, 1e-12));
  }

  #[test]
  fn joe_theta_from_tau_zero_tau_gives_independence() {
    let c = Joe::new();
    assert!(approx(c.theta_from_tau(0.0).unwrap(), 1.0, 1e-9));
  }

  #[test]
  fn joe_theta_from_tau_inverts_tau_via_series() {
    let c = Joe::new();
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
    let c = Joe::new();
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
    let c = Joe::new();
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
    let mut c = Joe::new();
    c.set_theta(2.0);
    let x = array![[0.25_f64, 0.75], [0.5, 0.5], [0.1, 0.9]];
    let pdf = c.pdf(x.view()).unwrap();
    for &p in pdf.iter() {
      assert!(p > 0.0 && p.is_finite(), "pdf={p}");
    }
  }

  #[test]
  fn joe_tail_dependence_matches_gumbel_form() {
    let mut c = Joe::new();
    c.set_theta(2.0);
    let td = c.tail_dependence();
    let expected = 2.0 - 2.0_f64.powf(0.5);
    assert!(approx(td.upper, expected, 1e-12), "got {}", td.upper);
    assert_eq!(td.lower, 0.0);
  }
}
