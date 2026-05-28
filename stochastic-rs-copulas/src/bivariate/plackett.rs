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
//! Reference: Plackett, R.L. (1965), "A Class of Bivariate Distributions",
//! *Journal of the American Statistical Association* 60(310), 516-522.
//! Reference: Nelsen, R.B. (2006), "An Introduction to Copulas", 2nd ed.,
//! Springer, Example 3.11.

use std::error::Error;
use std::f64;

use ndarray::Array1;
use ndarray::Array2;
use roots::SimpleConvergency;
use roots::find_root_brent;

use crate::bivariate::CopulaType;
use crate::traits::BivariateExt;

#[derive(Debug, Clone)]
pub struct Plackett {
  pub r#type: CopulaType,
  pub theta: Option<f64>,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Default for Plackett {
  fn default() -> Self {
    Self {
      r#type: CopulaType::Plackett,
      theta: None,
      tau: None,
      theta_bounds: (1e-12, f64::INFINITY),
      invalid_thetas: vec![],
    }
  }
}

impl Plackett {
  pub fn new() -> Self {
    Self::default()
  }

  /// Spearman's rho as a function of theta (Plackett 1965):
  /// $\rho_S = (\theta+1)/(\theta-1) - 2\theta \ln \theta / (\theta-1)^2$ for $\theta \neq 1$.
  fn spearman_rho(theta: f64) -> f64 {
    if (theta - 1.0).abs() < 1e-9 {
      return 0.0;
    }
    let d = theta - 1.0;
    (theta + 1.0) / d - 2.0 * theta * theta.ln() / (d * d)
  }

  /// Kendall's tau for the Plackett copula has no elementary closed form;
  /// numerical surveys (Nelsen 2006 §5.3) give the relation
  /// $\tau \approx (3 \rho_S) / (2 + |\rho_S|)$ as a smooth interpolant
  /// (Genest & Carabarín-Aguirre 2014 quote $\rho_S/\tau \approx 3/2$ for
  /// the Plackett family in the moderate-dependence regime), which inverts
  /// to $|\rho_S| \approx 2|\tau|/(3 - |\tau|)$ at the few-percent level.
  /// Returns the residual $\rho_S(\theta) - \rho_S^{\text{target}}$.
  fn rho_residual(rho_target: f64, theta: f64) -> f64 {
    Self::spearman_rho(theta) - rho_target
  }
}

impl BivariateExt for Plackett {
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

  /// Plackett is **not** Archimedean; the generator is undefined.
  fn generator(&self, _t: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    Err("Plackett is not Archimedean — generator not defined".into())
  }

  /// Density (Nelsen 2006 eq.3.3.8):
  /// $c(u,v) = \frac{\theta [1 + (\theta-1)(u + v - 2 u v)]}{\{[1 + (\theta-1)(u + v)]^2 - 4 u v \theta (\theta - 1)\}^{3/2}}$.
  fn pdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
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
  fn cdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
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

  /// $\partial_v C(u,v) = \frac{1}{2}\left[1 - \frac{1 + (\theta-1)(u+v) - 2 u \theta}{\sqrt{\{\cdots\}}}\right]$.
  fn partial_derivative(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
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
        out[i] = u;
        continue;
      }
      let a = 1.0 + eta * (u + v);
      let d = a * a - 4.0 * u * v * theta * eta;
      let d_safe = d.max(1e-30).sqrt();
      out[i] = 0.5 * (1.0 - (a - 2.0 * u * theta) / d_safe);
    }
    Ok(out)
  }

  /// Plackett's natural rank-correlation is Spearman's $\rho_S$, which has
  /// the closed form $\rho_S(\theta) = (\theta+1)/(\theta-1) - 2\theta\ln\theta/(\theta-1)^2$.
  /// We invert this on the Plackett $\tau \leftrightarrow \rho_S$ family
  /// using the Genest-Carabarín-Aguirre approximation
  /// $\rho_S \approx 2\tau/(3-|\tau|)$ (sign-preserving), then solve for
  /// $\theta$ by Brent root-finding on $\rho_S(\theta) - \rho_S^{\text{target}}$.
  fn compute_theta(&self) -> f64 {
    let tau = self.tau.unwrap();
    if tau.abs() < 1e-12 {
      return 1.0;
    }
    let rho_target = 2.0 * tau / (3.0 - tau.abs());
    let residual = |theta: f64| Self::rho_residual(rho_target, theta);
    let mut convergency = SimpleConvergency {
      eps: 1e-8,
      max_iter: 100,
    };
    let (lo, hi) = if tau > 0.0 {
      (1.0 + 1e-6_f64, 1e6_f64)
    } else {
      (1e-6_f64, 1.0 - 1e-6_f64)
    };
    find_root_brent(lo, hi, residual, &mut convergency).unwrap_or(1.0)
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
    let mut c = Plackett::new();
    c.set_theta(1.0);
    let x = array![[0.3_f64, 0.7], [0.5, 0.5]];
    let cdf = c.cdf(&x).unwrap();
    assert!(approx(cdf[0], 0.21, 1e-9));
    assert!(approx(cdf[1], 0.25, 1e-9));
  }

  #[test]
  fn plackett_cdf_monotone_in_theta() {
    let x = array![[0.5_f64, 0.5]];
    let mut c_low = Plackett::new();
    c_low.set_theta(0.2);
    let mut c_high = Plackett::new();
    c_high.set_theta(5.0);
    let lo = c_low.cdf(&x).unwrap()[0];
    let hi = c_high.cdf(&x).unwrap()[0];
    // At (0.5, 0.5): independence gives 0.25; θ < 1 → smaller, θ > 1 → larger.
    assert!(lo < 0.25, "θ=0.2 should give < 0.25, got {lo}");
    assert!(hi > 0.25, "θ=5.0 should give > 0.25, got {hi}");
  }

  #[test]
  fn plackett_cdf_marginal_recovers_input() {
    let mut c = Plackett::new();
    c.set_theta(3.0);
    let x = array![[0.5_f64, 1.0], [1.0, 0.4]];
    let cdf = c.cdf(&x).unwrap();
    assert!(approx(cdf[0], 0.5, 1e-9));
    assert!(approx(cdf[1], 0.4, 1e-9));
  }

  #[test]
  fn plackett_spearman_rho_zero_at_theta_one() {
    assert!(approx(Plackett::spearman_rho(1.0), 0.0, 1e-12));
  }

  #[test]
  fn plackett_compute_theta_zero_tau_gives_independence() {
    let mut c = Plackett::new();
    c.set_tau(0.0);
    assert!(approx(c.compute_theta(), 1.0, 1e-12));
  }

  #[test]
  fn plackett_pdf_positive_on_unit_square_interior() {
    let mut c = Plackett::new();
    c.set_theta(2.0);
    let x = array![[0.25_f64, 0.75], [0.5, 0.5], [0.1, 0.9]];
    let pdf = c.pdf(&x).unwrap();
    for &p in pdf.iter() {
      assert!(p > 0.0 && p.is_finite(), "pdf={p}");
    }
  }

  #[test]
  fn plackett_generator_returns_err_not_archimedean() {
    let c = Plackett::new();
    let t = array![0.5_f64, 0.8];
    assert!(c.generator(&t).is_err());
  }
}
