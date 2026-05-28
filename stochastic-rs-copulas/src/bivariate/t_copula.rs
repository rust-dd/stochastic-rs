//! # Bivariate Student-t copula
//!
//! $$
//! C_{\rho,\nu}(u,v) = T_{\rho,\nu}\!\big(t_\nu^{-1}(u),\, t_\nu^{-1}(v)\big),
//! $$
//! where $T_{\rho,\nu}$ is the bivariate Student-t CDF with correlation
//! $\rho \in (-1, 1)$ and degrees of freedom $\nu > 0$, and $t_\nu^{-1}$
//! the univariate Student-t quantile.
//!
//! - **Kendall's tau:** $\tau = \tfrac{2}{\pi}\arcsin\rho$ (same as Gaussian).
//! - **Tail dependence:** $\lambda_U = \lambda_L = 2\, t_{\nu+1}\!\big(
//!   -\sqrt{(\nu+1)(1-\rho)/(1+\rho)}\big)$ — symmetric and strictly
//!   positive for finite $\nu$, in contrast to Gaussian.
//!
//! In the limit $\nu \to \infty$ the t-copula collapses to the Gaussian
//! copula.
//!
//! The bivariate CDF is evaluated through the Dunnett-Sobel (1955) 1D
//! reduction
//! $$
//! T_{\rho,\nu}(h, k) = \int_{-\infty}^{h} t_\nu(s)\, T_{\nu+1}\!\Bigg(
//! \frac{k - \rho s}{\sqrt{1-\rho^2}} \sqrt{\frac{\nu+1}{\nu+s^2}}
//! \Bigg) ds,
//! $$
//! with the substitution $u = F_\nu(s)$ collapsing the half-infinite range
//! to $[0, F_\nu(h)]$; Gauss-Legendre on 64 nodes delivers $\sim
//! 10^{-10}$ accuracy across $\nu \in [2, 30]$.
//!
//! Reference: Embrechts, P., Lindskog, F., McNeil, A.J. (2003),
//! "Modelling Dependence with Copulas and Applications to Risk
//! Management", in *Handbook of Heavy Tailed Distributions in Finance*,
//! Elsevier, ch. 8.
//! Reference: Dunnett, C.W., Sobel, M. (1955), "Approximations to the
//! probability integral and certain percentage points of a multivariate
//! analogue of Student's t-distribution", *Biometrika* 42(1/2), 258-260.

use std::error::Error;
use std::f64;

use gauss_quad::GaussLegendre;
use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_distributions::special::beta_i;
use stochastic_rs_distributions::special::ln_gamma;
use stochastic_rs_distributions::special::ndtri;

use crate::bivariate::CopulaType;
use crate::traits::BivariateExt;

#[derive(Debug, Clone)]
pub struct TCopula {
  pub r#type: CopulaType,
  /// Correlation $\rho \in (-1, 1)$, stored under the trait's `theta`
  /// field for single-parameter compatibility. Set degrees of freedom
  /// `nu` via [`TCopula::with_nu`].
  pub theta: Option<f64>,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
  /// Degrees of freedom $\nu > 0$. Default 4.
  pub nu: f64,
}

impl Default for TCopula {
  fn default() -> Self {
    Self {
      r#type: CopulaType::TCopula,
      theta: None,
      tau: None,
      theta_bounds: (-1.0, 1.0),
      invalid_thetas: vec![],
      nu: 4.0,
    }
  }
}

impl TCopula {
  pub fn new() -> Self {
    Self::default()
  }

  /// Construct with explicit degrees of freedom.
  pub fn with_nu(nu: f64) -> Self {
    assert!(nu > 0.0, "nu must be positive, got {nu}");
    Self {
      nu,
      ..Self::default()
    }
  }

  /// Standard Student-t density $f_\nu(x)$.
  fn t_pdf(x: f64, nu: f64) -> f64 {
    let log_norm =
      ln_gamma(0.5 * (nu + 1.0)) - 0.5 * (nu * std::f64::consts::PI).ln() - ln_gamma(0.5 * nu);
    let log_kernel = -0.5 * (nu + 1.0) * (1.0 + x * x / nu).ln();
    (log_norm + log_kernel).exp()
  }

  /// Standard Student-t CDF $F_\nu(x)$ via the regularised incomplete-beta
  /// identity $F_\nu(x) = 1 - \tfrac{1}{2} I_{\nu/(\nu+x^2)}(\nu/2, 1/2)$
  /// for $x \ge 0$.
  fn t_cdf(x: f64, nu: f64) -> f64 {
    if !x.is_finite() {
      return if x > 0.0 { 1.0 } else { 0.0 };
    }
    let t = nu / (nu + x * x);
    let half = 0.5 * beta_i(0.5 * nu, 0.5, t);
    if x >= 0.0 { 1.0 - half } else { half }
  }

  /// Quantile $t_\nu^{-1}(p)$: Cornish-Fisher-style normal seed refined by
  /// 40 Newton steps on `[0, 1]`. Identical to the routine in
  /// `stochastic_rs_distributions::studentt::SimdStudentT::inv_cdf`.
  fn t_quantile(p: f64, nu: f64) -> f64 {
    if p <= 0.0 {
      return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
      return f64::INFINITY;
    }
    let z = ndtri(p);
    let mut x = z * (1.0 + (z * z + 1.0) / (4.0 * nu));
    for _ in 0..40 {
      let cdf = Self::t_cdf(x, nu);
      let f = cdf - p;
      let pdf = Self::t_pdf(x, nu);
      if pdf <= 0.0 {
        break;
      }
      let dx = f / pdf;
      let new_x = x - dx;
      if (new_x - x).abs() < 1e-14 * (1.0 + x.abs()) {
        return new_x;
      }
      x = new_x;
    }
    x
  }

  /// Bivariate Student-t CDF $T_{\rho,\nu}(h, k)$ via Dunnett-Sobel
  /// 1D reduction with $u = F_\nu(s)$ change-of-variable.
  fn bivariate_t_cdf(h: f64, k: f64, rho: f64, nu: f64) -> f64 {
    // Degenerate marginals.
    let f_h = Self::t_cdf(h, nu);
    if f_h <= 0.0 {
      return 0.0;
    }
    if rho.abs() >= 1.0 - 1e-12 {
      return if rho > 0.0 {
        f_h.min(Self::t_cdf(k, nu))
      } else {
        (f_h + Self::t_cdf(k, nu) - 1.0).max(0.0)
      };
    }
    let one_minus_rho2 = 1.0 - rho * rho;
    let nu_plus_one = nu + 1.0;
    let sqrt_one_minus_rho2 = one_minus_rho2.sqrt();
    let quad = GaussLegendre::new(std::num::NonZeroUsize::new(64).unwrap());
    quad.integrate(0.0, f_h, |u| {
      let s = Self::t_quantile(u, nu);
      let scale = (nu_plus_one / (nu + s * s)).sqrt() / sqrt_one_minus_rho2;
      Self::t_cdf((k - rho * s) * scale, nu_plus_one)
    })
  }

  /// Bivariate Student-t density $f_{\rho,\nu}(x, y)$.
  fn bivariate_t_pdf(x: f64, y: f64, rho: f64, nu: f64) -> f64 {
    let one_minus_rho2 = 1.0 - rho * rho;
    let log_norm = ln_gamma(0.5 * (nu + 2.0))
      - ln_gamma(0.5 * nu)
      - (nu * std::f64::consts::PI).ln()
      - 0.5 * one_minus_rho2.ln();
    let q = (x * x - 2.0 * rho * x * y + y * y) / (nu * one_minus_rho2);
    let log_kernel = -0.5 * (nu + 2.0) * (1.0 + q).ln();
    (log_norm + log_kernel).exp()
  }
}

impl BivariateExt for TCopula {
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

  /// t-copula is **not** Archimedean — no scalar generator.
  fn generator(&self, _t: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    Err("t-copula is not Archimedean — generator not defined".into())
  }

  fn pdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;
    let rho = self.theta.unwrap();
    let nu = self.nu;
    let u_col = x.column(0);
    let v_col = x.column(1);
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i];
      if u <= 0.0 || u >= 1.0 || v <= 0.0 || v >= 1.0 {
        out[i] = 0.0;
        continue;
      }
      let xx = Self::t_quantile(u, nu);
      let yy = Self::t_quantile(v, nu);
      let num = Self::bivariate_t_pdf(xx, yy, rho, nu);
      let den = Self::t_pdf(xx, nu) * Self::t_pdf(yy, nu);
      out[i] = num / den;
    }
    Ok(out)
  }

  fn cdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;
    let rho = self.theta.unwrap();
    let nu = self.nu;
    let u_col = x.column(0);
    let v_col = x.column(1);
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
      let xx = Self::t_quantile(u, nu);
      let yy = Self::t_quantile(v, nu);
      out[i] = Self::bivariate_t_cdf(xx, yy, rho, nu);
    }
    Ok(out)
  }

  /// $\partial_v C(u,v) = T_{\nu+1}\!\Big(\frac{x - \rho y}{\sqrt{1-\rho^2}}
  /// \sqrt{\frac{\nu+1}{\nu+y^2}}\Big)$ where $x = t_\nu^{-1}(u),\,
  /// y = t_\nu^{-1}(v)$.
  fn partial_derivative(&self, x: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;
    let rho = self.theta.unwrap();
    let nu = self.nu;
    let one_minus_rho2 = 1.0 - rho * rho;
    let sqrt_one_minus_rho2 = one_minus_rho2.sqrt();
    let nu_plus_one = nu + 1.0;
    let u_col = x.column(0);
    let v_col = x.column(1);
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
      let xx = Self::t_quantile(u, nu);
      let yy = Self::t_quantile(v, nu);
      let scale = (nu_plus_one / (nu + yy * yy)).sqrt() / sqrt_one_minus_rho2;
      out[i] = Self::t_cdf((xx - rho * yy) * scale, nu_plus_one);
    }
    Ok(out)
  }

  /// Closed-form Kendall's tau inversion $\rho = \sin(\pi\tau/2)$.
  fn compute_theta(&self) -> f64 {
    let tau = self.tau.unwrap();
    (0.5 * std::f64::consts::PI * tau).sin().clamp(-1.0, 1.0)
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
  fn t_cdf_marginal_recovers_input() {
    let mut c = TCopula::with_nu(4.0);
    c.set_theta(0.5);
    let x = array![[0.4_f64, 1.0], [1.0, 0.7]];
    let cdf = c.cdf(&x).unwrap();
    assert!(approx(cdf[0], 0.4, 1e-6));
    assert!(approx(cdf[1], 0.7, 1e-6));
  }

  #[test]
  fn t_cdf_zero_rho_at_origin_is_one_quarter() {
    // For bivariate Student-t with ρ = 0, the components are uncorrelated
    // but **not independent** (they share a common χ²_ν mixing scale).
    // Yet at the marginal medians (0, 0), the sign-decoupling representation
    //   (X, Y) = (Z₁, Z₂) / √(W/ν), Z ⊥ W,
    // gives sign(X) = sign(Z₁), sign(Y) = sign(Z₂), with Z₁ ⊥ Z₂. Hence
    //   P(X ≤ 0, Y ≤ 0) = P(Z₁ ≤ 0) · P(Z₂ ≤ 0) = 1/4 exactly.
    let mut c = TCopula::with_nu(4.0);
    c.set_theta(0.0);
    let x = array![[0.5_f64, 0.5]];
    let cdf = c.cdf(&x).unwrap();
    assert!(approx(cdf[0], 0.25, 1e-6), "got {}", cdf[0]);
  }

  #[test]
  fn t_compute_theta_matches_sin_formula() {
    let mut c = TCopula::with_nu(4.0);
    c.set_tau(0.25);
    let expected = (0.5_f64 * std::f64::consts::PI * 0.25).sin();
    assert!(approx(c.compute_theta(), expected, 1e-12));
  }

  #[test]
  fn t_pdf_symmetric_in_uv() {
    let mut c = TCopula::with_nu(5.0);
    c.set_theta(0.3);
    let x_ab = array![[0.3_f64, 0.7]];
    let x_ba = array![[0.7_f64, 0.3]];
    let pdf_ab = c.pdf(&x_ab).unwrap();
    let pdf_ba = c.pdf(&x_ba).unwrap();
    assert!(approx(pdf_ab[0], pdf_ba[0], 1e-9));
  }

  #[test]
  fn t_pdf_large_nu_approaches_gaussian_copula() {
    // At ν → ∞ the t-copula collapses to the Gaussian copula.
    // Spot-check density vs Gaussian-copula density at a fixed (u, v, ρ)
    // — use ν = 500.
    let mut c = TCopula::with_nu(500.0);
    c.set_theta(0.5);
    let x = array![[0.3_f64, 0.7]];
    let pdf_t = c.pdf(&x).unwrap();

    // Gaussian copula density at (0.3, 0.7), ρ=0.5:
    // c(u,v) = (1/√(1-ρ²)) · exp{(2ρ x y - ρ²(x²+y²))/(2(1-ρ²))}
    // with x = Φ⁻¹(u), y = Φ⁻¹(v).
    let rho = 0.5_f64;
    let xx = ndtri(0.3);
    let yy = ndtri(0.7);
    let r2 = rho * rho;
    let factor = (2.0 * rho * xx * yy - r2 * (xx * xx + yy * yy)) / (2.0 * (1.0 - r2));
    let pdf_gauss = factor.exp() / (1.0 - r2).sqrt();
    assert!(
      (pdf_t[0] - pdf_gauss).abs() < 0.01,
      "pdf_t={}, pdf_gauss={}",
      pdf_t[0],
      pdf_gauss
    );
  }

  #[test]
  fn t_cdf_symmetry_in_rho() {
    // C_{ρ,ν}(u, v) - u·v   should be an odd function of ρ at u = v = 1/2.
    // Test: C_{-ρ}(0.5, 0.5) + C_{ρ}(0.5, 0.5) = 2·C_{0}(0.5, 0.5).
    let mut c_pos = TCopula::with_nu(4.0);
    c_pos.set_theta(0.4);
    let mut c_neg = TCopula::with_nu(4.0);
    c_neg.set_theta(-0.4);
    let mut c_zero = TCopula::with_nu(4.0);
    c_zero.set_theta(0.0);
    let pt = array![[0.5_f64, 0.5]];
    let lhs = c_pos.cdf(&pt).unwrap()[0] + c_neg.cdf(&pt).unwrap()[0];
    let rhs = 2.0 * c_zero.cdf(&pt).unwrap()[0];
    assert!(approx(lhs, rhs, 1e-6), "symmetry: lhs = {lhs}, rhs = {rhs}");
  }

  #[test]
  fn t_partial_derivative_matches_finite_diff() {
    let mut c = TCopula::with_nu(4.0);
    c.set_theta(0.4);
    let u = 0.3_f64;
    let v = 0.6_f64;
    let h = 1e-4_f64;
    let pd = c.partial_derivative(&array![[u, v]]).unwrap()[0];
    let cdf_hi = c.cdf(&array![[u, v + h]]).unwrap()[0];
    let cdf_lo = c.cdf(&array![[u, v - h]]).unwrap()[0];
    let fd = (cdf_hi - cdf_lo) / (2.0 * h);
    assert!(
      approx(pd, fd, 1e-3),
      "analytic ∂C/∂v = {pd}, finite-diff = {fd}"
    );
  }
}
