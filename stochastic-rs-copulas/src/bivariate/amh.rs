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

use std::error::Error;
use std::f64;

use ndarray::Array1;
use ndarray::Array2;
use roots::SimpleConvergency;
use roots::find_root_brent;

use crate::bivariate::CopulaType;
use crate::traits::BivariateExt;

#[derive(Debug, Clone)]
pub struct Amh {
  pub r#type: CopulaType,
  pub theta: Option<f64>,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Default for Amh {
  fn default() -> Self {
    Self {
      r#type: CopulaType::Amh,
      theta: None,
      tau: None,
      theta_bounds: (-1.0, 1.0),
      invalid_thetas: vec![],
    }
  }
}

impl Amh {
  pub fn new() -> Self {
    Self::default()
  }

  /// Kendall's tau in closed form
  /// $$
  /// \tau(\theta) = 1 - \frac{2}{3\theta^2}\big[\theta + (1-\theta)^2 \ln(1-\theta)\big],
  /// $$
  /// continuous on $\theta \in [-1, 1)$ with $\lim_{\theta\to 0}\tau = 0$
  /// (a removable singularity handled by a fourth-order Taylor expansion).
  ///
  /// Returns the residual $\tau(\theta) - \tau_{\text{target}}$.
  fn tau_residual(tau_target: f64, theta: f64) -> f64 {
    let tau_theta = if theta.abs() < 1e-4 {
      // Series around θ = 0: τ = 2θ/9 + θ²/18 + 2θ³/75 + 5θ⁴/294 + O(θ⁵).
      let t2 = theta * theta;
      let t3 = t2 * theta;
      let t4 = t2 * t2;
      2.0 / 9.0 * theta + t2 / 18.0 + 2.0 / 75.0 * t3 + 5.0 / 294.0 * t4
    } else {
      let one_minus_theta = 1.0 - theta;
      let bracket = theta + one_minus_theta * one_minus_theta * one_minus_theta.ln();
      1.0 - 2.0 * bracket / (3.0 * theta * theta)
    };
    tau_theta - tau_target
  }
}

impl BivariateExt for Amh {
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

  /// Archimedean generator $\varphi(t) = \ln\frac{1 - \theta(1-t)}{t}$.
  fn generator(&self, t: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
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
      let d = 1.0 - theta * (1.0 - u) * (1.0 - v);
      let num = (1.0 - theta) * d + 2.0 * theta * u * v;
      out[i] = num / (d * d * d);
    }
    Ok(out)
  }

  /// CDF $C(u,v) = uv / [1 - \theta(1-u)(1-v)]$.
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
      out[i] = u * v / (1.0 - theta * (1.0 - u) * (1.0 - v));
    }
    Ok(out)
  }

  /// $\partial_u C(u,v) = v[1 - \theta(1-v)] / D^2$.
  fn partial_derivative(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;
    let u_col = x.column(0);
    let v_col = x.column(1);
    let theta = self.theta.unwrap();
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i];
      if v <= 0.0 {
        out[i] = 0.0;
        continue;
      }
      if v >= 1.0 {
        out[i] = 1.0;
        continue;
      }
      let d = 1.0 - theta * (1.0 - u) * (1.0 - v);
      out[i] = v * (1.0 - theta * (1.0 - v)) / (d * d);
    }
    Ok(out)
  }

  fn compute_theta(&self) -> f64 {
    let tau = self.tau.unwrap();
    // AMH τ is monotonically increasing on (-1, 1) with the bracket
    // [(5 - 8 ln 2)/3, 1/3]; saturate outside and run Brent inside.
    let tau_lo = (5.0 - 8.0 * 2.0_f64.ln()) / 3.0;
    let tau_hi = 1.0 / 3.0;
    if tau <= tau_lo {
      return -1.0;
    }
    if tau >= tau_hi {
      return 1.0 - 1e-9;
    }
    let residual = |theta: f64| Self::tau_residual(tau, theta);
    let mut convergency = SimpleConvergency {
      eps: 1e-10,
      max_iter: 100,
    };
    find_root_brent(-1.0 + 1e-9, 1.0 - 1e-9, residual, &mut convergency).unwrap_or(0.0)
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
  fn amh_independence_theta_zero_reproduces_product_cdf() {
    let mut c = Amh::new();
    c.set_theta(0.0);
    let x = array![[0.3_f64, 0.7], [0.5, 0.5], [0.9, 0.1]];
    let cdf = c.cdf(&x).unwrap();
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
    let pdf = c.pdf(&x).unwrap();
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
    let cdf = c.cdf(&x).unwrap();
    assert!(approx(cdf[0], 0.4, 1e-12));
    assert!(approx(cdf[1], 0.2, 1e-12));
  }

  #[test]
  fn amh_tau_residual_matches_paper_bounds() {
    // τ → -0.1817 as θ → -1; τ → 1/3 as θ → 1⁻.
    let res_lo = Amh::tau_residual(0.0, -0.999_999);
    let res_hi = Amh::tau_residual(0.0, 0.999_999);
    // tau_residual(0, θ) = τ(θ); compare to paper bounds.
    let tau_lo = (5.0 - 8.0 * 2.0_f64.ln()) / 3.0;
    assert!(
      approx(res_lo, tau_lo, 1e-3),
      "τ(-1) ≈ {tau_lo}, got {res_lo}"
    );
    assert!(approx(res_hi, 1.0 / 3.0, 1e-3), "τ(1) ≈ 1/3, got {res_hi}");
  }

  #[test]
  fn amh_compute_theta_inverts_tau_within_bracket() {
    // Pick a θ inside the bracket, compute τ, then invert and recover.
    let target_theta = 0.6_f64;
    let tau = Amh::tau_residual(0.0, target_theta);
    let mut c = Amh::new();
    c.set_tau(tau);
    let recovered = c.compute_theta();
    assert!(
      approx(recovered, target_theta, 1e-6),
      "expected {target_theta}, got {recovered}"
    );
  }

  #[test]
  fn amh_tau_zero_recovers_independence() {
    let mut c = Amh::new();
    c.set_tau(0.0);
    let theta = c.compute_theta();
    assert!(approx(theta, 0.0, 1e-6), "τ=0 should give θ=0, got {theta}");
  }
}
