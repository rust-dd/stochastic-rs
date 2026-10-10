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
//! Reference: Aas, K., Czado, C., Frigessi, A., Bakken, H. (2009), "Pair-copula constructions of multiple dependence", *Insurance: Mathematics and Economics* 44(2), 182-198, App. B.2 (the h-function inverse), DOI 10.1016/j.insmatheco.2007.02.001.

use gauss_quad::GaussLegendre;
use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_distributions::DistributionExt;
use stochastic_rs_distributions::special::ln_gamma;
use stochastic_rs_distributions::studentt::SimdStudentT;

use crate::bivariate::CopulaType;
use crate::bivariate::conditional::conditional_cdf;
use crate::bivariate::conditional::conditional_quantiles;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;

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
  /// Degrees of freedom $\nu > 0$. Default 4. Private and validated — read
  /// via [`TCopula::nu`], write via [`TCopula::set_nu`].
  nu: f64,
}

/// The degrees-of-freedom rule of both t-copulas: `0 < nu < ∞`, and `nu / 2 > 0`, which only `5e-324` fails and which
/// `SimdStudentT::new` and `SimdChiSquared::new` assert, so a value they would panic on is an error up front.
pub(crate) fn check_nu(nu: f64) -> Result<(), CopulaError> {
  let constraint = if !nu.is_finite() || nu <= 0.0 {
    "0 < nu < ∞"
  } else if nu * 0.5 == 0.0 {
    "nu / 2 > 0"
  } else {
    return Ok(());
  };
  Err(CopulaError::InvalidParameter {
    name: "nu",
    value: nu,
    constraint: constraint.into(),
  })
}

impl Default for TCopula {
  fn default() -> Self {
    Self {
      r#type: CopulaType::TCopula,
      theta: None,
      tau: None,
      theta_bounds: (-1.0, 1.0),
      // Mirrors `GaussianCopula`'s own `invalid_thetas`: `rho = ±1` divides
      // by `sqrt(1 - rho^2)` in `pdf` (and, less directly, in
      // `partial_derivative`/`percent_point`), so it is rejected the same
      // way here rather than silently producing `NaN`.
      invalid_thetas: vec![-1.0, 1.0],
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
    let mut c = Self::default();
    if let Err(e) = c.set_nu(nu) {
      panic!("{e}");
    }
    c
  }

  /// Current degrees of freedom $\nu$.
  pub fn nu(&self) -> f64 {
    self.nu
  }

  /// Override the degrees of freedom; an error under the rule of `check_nu`, as for `TMultivariate::set_nu`.
  pub fn set_nu(&mut self, nu: f64) -> Result<(), CopulaError> {
    check_nu(nu)?;
    self.nu = nu;
    Ok(())
  }

  /// `u = t_ν(t_{ν+1}⁻¹(y)√((ν + x²)(1 − ρ²)/(ν + 1)) + ρx)`, `x = t_ν⁻¹(v)`, and the two-point law the `v → 0, 1`
  /// limit of `∂_v C` leaves at the edges.
  pub(crate) fn inverse(rho: f64, nu: f64) -> impl Fn(f64, f64) -> f64 {
    let one_minus_rho2 = 1.0 - rho * rho;
    let edge = rho * ((nu + 1.0) / one_minus_rho2).sqrt();
    move |y, v| {
      let x = Self::t_quantile(v, nu);
      let q = Self::t_quantile(y, nu + 1.0);
      if x.is_infinite() {
        return if q <= -x.signum() * edge { 0.0 } else { 1.0 };
      }
      let scale = ((nu + x * x) * one_minus_rho2 / (nu + 1.0)).sqrt();
      Self::t_cdf(q * scale + rho * x, nu)
    }
  }

  /// The `t_ν` density of `stochastic_rs_distributions`, whose `DistributionExt` answers `Some` for every argument.
  fn t_pdf(x: f64, nu: f64) -> f64 {
    SimdStudentT::new(nu)
      .pdf(x)
      .expect("the Student-t law has a pdf")
  }

  /// The `t_ν` cdf of `stochastic_rs_distributions`, `1 − I_{ν/(ν+x²)}(ν/2, 1/2)/2` for `x ≥ 0`.
  fn t_cdf(x: f64, nu: f64) -> f64 {
    SimdStudentT::new(nu)
      .cdf(x)
      .expect("the Student-t law has a cdf")
  }

  /// The `t_ν` quantile of `stochastic_rs_distributions`: `∓∞` at `p ≤ 0` / `p ≥ 1`, Newton steps inside.
  fn t_quantile(p: f64, nu: f64) -> f64 {
    SimdStudentT::new(nu)
      .quantile(p)
      .expect("the Student-t law has a quantile")
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

  fn pdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, CopulaError> {
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

  fn cdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, CopulaError> {
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

  /// `∂_v C = t_{ν+1}((x − ρy)√((ν + 1)/((ν + y²)(1 − ρ²))))`, `x, y = t_ν⁻¹(u), t_ν⁻¹(v)`, whose `v → 0, 1` limit
  /// `t_{ν+1}(±ρ√((ν + 1)/(1 − ρ²)))` holds for every `u`; NaN for `v` outside `[0, 1]`.
  fn partial_derivative(&self, x: &Array2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let rho = self.theta.unwrap();
    let nu = self.nu;
    let one_minus_rho2 = 1.0 - rho * rho;
    let sqrt_one_minus_rho2 = one_minus_rho2.sqrt();
    let nu_plus_one = nu + 1.0;
    let boundary_arg = rho * nu_plus_one.sqrt() / sqrt_one_minus_rho2;
    Ok(conditional_cdf(x, |u, v| {
      if v <= 0.0 {
        return Self::t_cdf(boundary_arg, nu_plus_one);
      }
      if v >= 1.0 {
        return Self::t_cdf(-boundary_arg, nu_plus_one);
      }
      let xx = Self::t_quantile(u, nu);
      let yy = Self::t_quantile(v, nu);
      let scale = (nu_plus_one / (nu + yy * yy)).sqrt() / sqrt_one_minus_rho2;
      Self::t_cdf((xx - rho * yy) * scale, nu_plus_one)
    }))
  }

  /// `TCopula::inverse` (Aas et al. App. B.2); NaN for `y` or `v` outside `[0, 1]`.
  fn percent_point(&self, y: &Array1<f64>, v: &Array1<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    conditional_quantiles(y, v, Self::inverse(self.theta.unwrap(), self.nu))
  }

  /// Closed-form Kendall's tau inversion $\rho = \sin(\pi\tau/2)$.
  fn compute_theta(&self) -> f64 {
    let tau = self.tau.unwrap();
    (0.5 * std::f64::consts::PI * tau).sin().clamp(-1.0, 1.0)
  }

  /// Symmetric tail dependence $\lambda_L = \lambda_U = 2\,t_{\nu+1}\!\big(
  /// -\sqrt{(\nu+1)(1-\rho)/(1+\rho)}\big)$, strictly positive for every
  /// finite $\nu$ and $\rho \in (-1,1)$ — in contrast to the Gaussian
  /// copula's asymptotic independence
  /// ([`crate::bivariate::gaussian::GaussianCopula`]).
  /// Reference: Embrechts, Lindskog, McNeil (2003), §3.2 (module header).
  fn tail_dependence(&self) -> TailDependence<f64> {
    self.assert_theta_valid_for_tail_dependence();
    let rho = self.theta.unwrap();
    let nu = self.nu;
    let arg = -(((nu + 1.0) * (1.0 - rho) / (1.0 + rho)).sqrt());
    let lambda = 2.0 * Self::t_cdf(arg, nu + 1.0);
    TailDependence {
      lower: lambda,
      upper: lambda,
    }
  }
}

#[cfg(test)]
mod tests;
