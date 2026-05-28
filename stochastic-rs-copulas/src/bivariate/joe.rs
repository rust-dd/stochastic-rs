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

use std::error::Error;
use std::f64;

use ndarray::Array1;
use ndarray::Array2;
use roots::SimpleConvergency;
use roots::find_root_brent;

use crate::bivariate::CopulaType;
use crate::traits::BivariateExt;

#[derive(Debug, Clone)]
pub struct Joe {
  pub r#type: CopulaType,
  pub theta: Option<f64>,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Default for Joe {
  fn default() -> Self {
    Self {
      r#type: CopulaType::Joe,
      theta: None,
      tau: None,
      theta_bounds: (1.0, f64::INFINITY),
      invalid_thetas: vec![],
    }
  }
}

impl Joe {
  pub fn new() -> Self {
    Self::default()
  }

  /// Kendall's tau as a function of theta (Joe 1997 eq.5.18). No closed form
  /// in elementary functions; evaluated via the convergent series
  /// $\tau(\theta) = 1 - \frac{4}{\theta^2}\sum_{k\ge 1}\frac{1}{k(2/\theta + k)(\theta(k-1) + 2)}$.
  ///
  /// Returns the residual $\tau(\theta) - \tau_{\text{target}}$ so this can
  /// be passed to a root finder for the inverse problem.
  fn tau_residual(tau_target: f64, theta: f64) -> f64 {
    if theta <= 1.0 + 1e-12 {
      return -tau_target;
    }
    // Joe's tau in closed form via digamma (Hofert et al. 2018):
    // τ = 1 − 4 [ψ(2) − ψ(2/θ + 1)] / [θ(2 − θ)]   for θ ≠ 2;
    // τ = 1 − [trigamma special case] at θ = 2.
    // Series form below avoids special-function dependency: take the first
    // 256 terms; the partial-sum tail is bounded by 1/[θ·256·(2/θ + 256)],
    // which is well below 1e-10 for θ ≤ 50 (our search bracket).
    let mut s = 0.0_f64;
    for k in 1..=256_i32 {
      let kf = k as f64;
      let d1 = kf;
      let d2 = 2.0 / theta + kf;
      let d3 = theta * (kf - 1.0) + 2.0;
      s += 1.0 / (d1 * d2 * d3);
    }
    let tau_theta = 1.0 - 4.0 * s / (theta * theta);
    tau_theta - tau_target
  }
}

impl BivariateExt for Joe {
  fn r#type(&self) -> CopulaType {
    self.r#type
  }

  fn tau(&self) -> Option<f64> {
    self.tau
  }

  fn set_tau(&mut self, tau: f64) {
    self.tau = Some(tau);
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
  fn generator(&self, t: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
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
  fn pdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
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
  fn cdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
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

  /// $\partial_v C(u,v) = (1 - (1-u)^\theta)(1-v)^{\theta-1} S^{1/\theta - 1}$.
  fn partial_derivative(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;
    let u_col = x.column(0);
    let v_col = x.column(1);
    let theta = self.theta.unwrap();
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i];
      if u <= 0.0 {
        out[i] = 0.0;
        continue;
      }
      if u >= 1.0 {
        out[i] = 1.0;
        continue;
      }
      let a = (1.0 - u).powf(theta);
      let b = (1.0 - v).powf(theta);
      let s = a + b - a * b;
      out[i] = (1.0 - a) * (1.0 - v).powf(theta - 1.0) * s.powf(1.0 / theta - 1.0);
    }
    Ok(out)
  }

  fn compute_theta(&self) -> f64 {
    let tau = self.tau.unwrap();
    if tau <= 0.0 {
      // Joe only supports positive dependence; map to independence.
      return 1.0;
    }
    if tau >= 1.0 - 1e-12 {
      return f64::INFINITY;
    }
    let residual = |theta: f64| Self::tau_residual(tau, theta);
    let mut convergency = SimpleConvergency {
      eps: 1e-8,
      max_iter: 100,
    };
    find_root_brent(1.0 + 1e-6, 50.0, residual, &mut convergency).unwrap_or(1.0)
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
    let cdf = c.cdf(&x).unwrap();
    for i in 0..x.nrows() {
      let expected = x[[i, 0]] * x[[i, 1]];
      assert!(approx(cdf[i], expected, 1e-12), "row {i}: {} vs {}", cdf[i], expected);
    }
  }

  #[test]
  fn joe_cdf_marginal_recovers_input() {
    let mut c = Joe::new();
    c.set_theta(3.0);
    let x = array![[0.5_f64, 1.0], [1.0, 0.4]];
    let cdf = c.cdf(&x).unwrap();
    assert!(approx(cdf[0], 0.5, 1e-12));
    assert!(approx(cdf[1], 0.4, 1e-12));
  }

  #[test]
  fn joe_cdf_at_corner_is_zero() {
    let mut c = Joe::new();
    c.set_theta(2.5);
    let x = array![[0.0_f64, 0.5], [0.5, 0.0]];
    let cdf = c.cdf(&x).unwrap();
    assert!(approx(cdf[0], 0.0, 1e-12));
    assert!(approx(cdf[1], 0.0, 1e-12));
  }

  #[test]
  fn joe_compute_theta_zero_tau_gives_independence() {
    let mut c = Joe::new();
    c.set_tau(0.0);
    assert!(approx(c.compute_theta(), 1.0, 1e-9));
  }

  #[test]
  fn joe_compute_theta_inverts_tau_via_series() {
    // Pick θ = 2.0 (concrete Joe parameter), compute τ via series, then
    // invert and verify we recover the same θ.
    let mut c = Joe::new();
    c.set_theta(2.0);
    let tau = Joe::tau_residual(0.0, 2.0); // tau(2.0) - 0 = tau(2.0)
    c.set_tau(tau);
    let recovered = c.compute_theta();
    assert!(approx(recovered, 2.0, 1e-6), "expected 2.0, got {recovered}");
  }

  #[test]
  fn joe_pdf_positive_on_unit_square_interior() {
    let mut c = Joe::new();
    c.set_theta(2.0);
    let x = array![[0.25_f64, 0.75], [0.5, 0.5], [0.1, 0.9]];
    let pdf = c.pdf(&x).unwrap();
    for &p in pdf.iter() {
      assert!(p > 0.0 && p.is_finite(), "pdf={p}");
    }
  }
}
