//! # Bivariate Student-t copula
//!
//! $$
//! C_{\rho,\nu}(u,v) = T_{\rho,\nu}\!\big(t_\nu^{-1}(u),\, t_\nu^{-1}(v)\big),
//! $$
//! $T_{\rho,\nu}$ is the bivariate Student-t cdf of correlation $\rho$, $t_\nu^{-1}$ the quantile;
//! $\nu \to \infty$ gives the Gaussian copula, and `cdf.rs` evaluates $T_{\rho,\nu}$.
//!
//! Reference: Embrechts, P., Lindskog, F., McNeil, A.J. (2003),
//! "Modelling Dependence with Copulas and Applications to Risk
//! Management", in *Handbook of Heavy Tailed Distributions in Finance*,
//! Elsevier, ch. 8.
//! Reference: Aas, K., Czado, C., Frigessi, A., Bakken, H. (2009), "Pair-copula constructions of multiple dependence", *Insurance: Mathematics and Economics* 44(2), 182-198, App. B.2 (the h-function inverse), DOI 10.1016/j.insmatheco.2007.02.001.

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;

use crate::bivariate::BivariateFamily;
use crate::bivariate::conditional::conditional_cdf;
use crate::bivariate::conditional::conditional_quantiles;
use crate::bivariate::conditional::copula_cdf;
use crate::bivariate::conditional::copula_pdf;
use crate::bivariate::conditional::on_open_square;
use crate::bivariate::gaussian::GaussianCopula;
use crate::bivariate::t_copula::cdf::BivariateT;
use crate::error::CopulaError;
use crate::student_t;
use crate::student_t::Law;
use crate::student_t::NORMAL_ABOVE;
use crate::student_t::TDensity;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::admitted;
use crate::traits::bivariate::checked_percent_point;
use crate::traits::bivariate::tau_out_of_range;

/// Below this a quantile's square, `ν` added, stays finite: past it the h-function and its inverse take logarithms.
const LARGE: f64 = 1e150;

#[derive(Debug, Clone)]
pub struct TCopula {
  theta: Option<f64>,
  nu: f64,
}

impl Default for TCopula {
  fn default() -> Self {
    Self {
      theta: None,
      nu: 4.0,
    }
  }
}

impl TCopula {
  /// A t copula with correlation `rho` and `nu` degrees of freedom; panics unless `-1 < rho < 1`
  /// and `nu` passes `check_nu` (`0 < nu < ∞`, `nu / 2 > 0`).
  pub fn new(rho: f64, nu: f64) -> Self {
    Self::try_new(rho, nu).unwrap_or_else(|e| panic!("{e}"))
  }

  /// `new` with the domain error returned instead of a panic.
  pub fn try_new(rho: f64, nu: f64) -> Result<Self, CopulaError> {
    student_t::check_nu(nu)?;
    let copula = Self {
      theta: Some(rho),
      nu,
    };
    copula.check_theta()?;
    Ok(copula)
  }

  /// The t copula whose Kendall's τ is `tau`, at `nu` degrees of freedom.
  pub fn from_tau(tau: f64, nu: f64) -> Result<Self, CopulaError> {
    Self::try_new(Self::default().theta_from_tau(tau)?, nu)
  }

  pub fn nu(&self) -> f64 {
    self.nu
  }

  /// Panics under the rule of `check_nu`: unless `0 < nu < ∞` and `nu / 2 > 0`.
  pub fn set_nu(&mut self, nu: f64) {
    student_t::check_nu(nu).unwrap_or_else(|e| panic!("{e}"));
    self.nu = nu;
  }

  /// `u = t_ν(t_{ν+1}⁻¹(y)√((ν + x²)(1 − ρ²)/(ν + 1)) + ρx)`, `x = t_ν⁻¹(v)`, in logarithms where a quantile passes
  /// `LARGE`, and the two-point law the `v → 0, 1` limit of `∂_v C` leaves at the edges.
  pub(crate) fn inverse(rho: f64, nu: f64) -> impl Fn(f64, f64) -> f64 {
    let (law, wide) = (Law::new(nu), Law::new(nu + 1.0));
    let one_minus_rho2 = (1.0 - rho) * (1.0 + rho);
    let edge = rho * ((nu + 1.0) / one_minus_rho2).sqrt();
    let ln_spread = 0.5 * (one_minus_rho2 / (nu + 1.0)).ln();
    move |y, v| {
      let (x, q) = (law.quantile(v), wide.quantile(y));
      if v <= 0.0 || v >= 1.0 {
        return if y < 1.0 && q.value <= -x.value.signum() * edge {
          0.0
        } else {
          1.0
        };
      }
      if x.value.abs() < LARGE && q.value.is_finite() {
        let scale = ((nu + x.value * x.value) * one_minus_rho2 / (nu + 1.0)).sqrt();
        return law.cdf_at(q.value * scale + rho * x.value);
      }
      if x.value.abs() < LARGE {
        let ln_scale = 0.5 * (nu + x.value * x.value).ln() + ln_spread;
        return law.cdf(q.value.signum(), q.ln_abs() + ln_scale);
      }
      // `B = xw`, `w = ρ + sgn(x) q √((1 + ν/x²)(1 − ρ²)/(ν + 1))`, whose `ν/x²` is below every rounding unit here.
      let (sign_w, ln_w) = if q.value.is_finite() {
        let w = rho + x.value.signum() * q.value * ln_spread.exp();
        (w.signum(), w.abs().ln())
      } else {
        (x.value.signum() * q.value.signum(), q.ln_abs() + ln_spread)
      };
      law.cdf_scaled(x, x.value.signum() * sign_w, ln_w)
    }
  }

  /// `∂_v C = t_{ν+1}((x − ρy)√((ν + 1)/((ν + y²)(1 − ρ²))))`, `x, y = t_ν⁻¹(u), t_ν⁻¹(v)`, in logarithms where a quantile
  /// passes `LARGE`, and its `v → 0, 1` limit `t_{ν+1}(±ρ√((ν + 1)/(1 − ρ²)))` at the edges.
  pub(crate) fn h(rho: f64, nu: f64) -> impl Fn(f64, f64) -> f64 {
    let (law, wide) = (Law::new(nu), Law::new(nu + 1.0));
    let sqrt_one_minus_rho2 = ((1.0 - rho) * (1.0 + rho)).sqrt();
    let nu_plus_one = nu + 1.0;
    let boundary_arg = rho * nu_plus_one.sqrt() / sqrt_one_minus_rho2;
    let reach = nu_plus_one.sqrt() / sqrt_one_minus_rho2;
    move |u, v| {
      if u >= 1.0 {
        return 1.0;
      }
      if v <= 0.0 || v >= 1.0 {
        return wide.cdf_at(if v <= 0.0 {
          boundary_arg
        } else {
          -boundary_arg
        });
      }
      if u <= 0.0 {
        return 0.0;
      }
      let (x, y) = (law.quantile(u), law.quantile(v));
      if x.value.is_finite() && y.value.abs() < LARGE {
        let gap = x.value - rho * y.value;
        // At the median a tiny `ν` overflows the scale: the argument is 0 there.
        if gap == 0.0 {
          return wide.cdf_at(0.0);
        }
        let scale = (nu_plus_one / (nu + y.value * y.value)).sqrt() / sqrt_one_minus_rho2;
        return wide.cdf_at(gap * scale);
      }
      // `x/√(ν + y²)` and `y/√(ν + y²)` from the logarithms; past `f64::MAX` the first carries the argument alone.
      let (ln_x_over_root, y_over_root) = if y.value.abs() < LARGE {
        let root = (nu + y.value * y.value).sqrt();
        (x.ln_abs() - root.ln(), y.value / root)
      } else {
        let lift = 0.5 * (nu * (-2.0 * y.ln_abs()).exp()).ln_1p();
        (law.ln_ratio(x, y) - lift, y.value.signum() * (-lift).exp())
      };
      let ratio = x.value.signum() * ln_x_over_root.exp();
      if ratio.is_finite() {
        wide.cdf_at((ratio - rho * y_over_root) * reach)
      } else {
        wide.cdf(x.value.signum(), ln_x_over_root + reach.ln())
      }
    }
  }

  /// `ln c` on the open square: the Gaussian copula's past `NORMAL_ABOVE`, else `TDensity::moderate` at the law's scores,
  /// else `student_t::ln_density` at their features, whose gap stays exact past `f64::MAX`.
  fn ln_density(&self) -> impl Fn(f64, f64) -> f64 {
    let (rho, nu) = (self.theta.unwrap(), self.nu);
    let (law, density) = (Law::new(nu), TDensity::new(rho, nu));
    move |u, v| {
      if nu >= NORMAL_ABOVE {
        return GaussianCopula::ln_density(rho, u, v);
      }
      let (sx, sy) = (law.quantile(u), law.quantile(v));
      if let Some(ln_c) = density.moderate(sx.value, sy.value) {
        return ln_c;
      }
      let (fx, fy) = (law.feature(sx), law.feature(sy));
      student_t::ln_density(&fx, &fy, law.abs_t_gap(&fx, &fy), rho, nu)
    }
  }

  /// `TCopula::inverse` (Aas et al. App. B.2).
  fn unchecked_inverse(
    &self,
    y: ArrayView1<f64>,
    v: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    Ok(conditional_quantiles(
      y,
      v,
      Self::inverse(self.theta.unwrap(), self.nu),
    ))
  }
}

impl BivariateExt for TCopula {
  fn family(&self) -> BivariateFamily {
    BivariateFamily::TCopula
  }

  fn theta(&self) -> Option<f64> {
    self.theta
  }

  fn set_theta(&mut self, theta: f64) {
    self.validate_theta(theta).unwrap_or_else(|e| panic!("{e}"));
    self.theta = Some(theta);
  }

  fn theta_bounds(&self) -> (f64, f64) {
    (-1.0, 1.0)
  }

  /// `ρ = ±1` divides by `√(1 − ρ²)`.
  fn invalid_thetas(&self) -> Vec<f64> {
    vec![-1.0, 1.0]
  }

  fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    copula_pdf(x, self.ln_density())
  }

  fn log_pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    on_open_square(x, self.ln_density())
  }

  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let bivariate = BivariateT::new(self.theta.unwrap(), self.nu);
    copula_cdf(x, |u, v| bivariate.copula_cdf(u, v))
  }

  /// `TCopula::h`.
  fn partial_derivative(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    conditional_cdf(x, Self::h(self.theta.unwrap(), self.nu))
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

  fn tau_from_theta(&self, theta: f64) -> f64 {
    2.0 / std::f64::consts::PI * theta.asin()
  }

  fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError> {
    if !(tau > -1.0 && tau < 1.0) {
      return Err(tau_out_of_range(tau, "-1.0 < tau < 1.0".into()));
    }
    // `ρ` rounds onto the excluded `±1` once `|τ| > 1 − 6e-9`.
    admitted(self, tau, (std::f64::consts::FRAC_PI_2 * tau).sin())
  }

  /// $$
  /// \lambda_L = \lambda_U = 2\,t_{\nu+1}\!\big(-\sqrt{(\nu+1)(1-\rho)/(1+\rho)}\big)
  /// $$
  /// Symmetric and positive for finite $\nu$; the Gaussian copula's are zero.
  /// Reference: Embrechts, Lindskog, McNeil (2003), §3.2 (module header).
  fn tail_dependence(&self) -> Result<TailDependence<f64>, CopulaError> {
    self.check_theta()?;
    let rho = self.theta.unwrap();
    let nu = self.nu;
    let arg = -(((nu + 1.0) * (1.0 - rho) / (1.0 + rho)).sqrt());
    let lambda = 2.0 * student_t::cdf(arg, nu + 1.0);
    Ok(TailDependence {
      lower: lambda,
      upper: lambda,
    })
  }
}

pub(crate) mod cdf;
mod dunnett_sobel;

#[cfg(test)]
mod tests;
