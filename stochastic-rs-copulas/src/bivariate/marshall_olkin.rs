//! # Marshall-Olkin (1967) bivariate copula
//!
//! $$
//! C_{\alpha,\beta}(u,v) =
//! \begin{cases}
//!   u^{1-\alpha}\, v & \text{if } u^\alpha \ge v^\beta,\\
//!   u\, v^{1-\beta} & \text{if } u^\alpha < v^\beta,
//! \end{cases}
//! \qquad \alpha, \beta \in (0, 1].
//! $$
//!
//! Non-exchangeable; the curve `u^α = v^β` carries a singular mass `αβ/(α + β − αβ)`, which is also Kendall's τ.
//! `theta` is the slice `α = β = θ`; [`MarshallOlkin::new`] takes the two shock rates; `λ_U = min(α, β)`.
//!
//! Reference: Marshall, A.W., Olkin, I. (1967), "A multivariate exponential distribution", *JASA* 62(317), 30-44, DOI 10.1080/01621459.1967.10482885.
//! Reference: Nelsen, R.B. (2006), "An Introduction to Copulas", 2nd ed., Springer, §3.1.1, eq. 3.1.3; §2.9 (the conditional-distribution method with a quasi-inverse), DOI 10.1007/0-387-28678-0.

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;

use crate::bivariate::BivariateFamily;
use crate::bivariate::conditional::conditional_cdf;
use crate::bivariate::conditional::conditional_quantiles;
use crate::bivariate::conditional::copula_cdf;
use crate::bivariate::conditional::on_open_square;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::admitted;
use crate::traits::bivariate::checked_percent_point;
use crate::traits::bivariate::tau_out_of_range;

#[derive(Debug, Clone, Default)]
pub struct MarshallOlkin {
  alpha: Option<f64>,
  beta: Option<f64>,
}

impl MarshallOlkin {
  /// `C_{α,β}`; panics unless `0 < alpha <= 1` and `0 < beta <= 1`.
  pub fn new(alpha: f64, beta: f64) -> Self {
    Self::try_new(alpha, beta).unwrap_or_else(|e| panic!("{e}"))
  }

  /// `new` with the domain error returned instead of a panic.
  pub fn try_new(alpha: f64, beta: f64) -> Result<Self, CopulaError> {
    let copula = Self {
      alpha: Some(alpha),
      beta: Some(beta),
    };
    copula.check_fit()?;
    Ok(copula)
  }

  /// The symmetric slice `α = β = 2τ/(1 + τ)`, whose Kendall's τ is `tau`.
  pub fn from_tau(tau: f64) -> Result<Self, CopulaError> {
    let theta = Self::default().theta_from_tau(tau)?;
    Self::try_new(theta, theta)
  }

  pub fn alpha(&self) -> Option<f64> {
    self.alpha
  }

  pub fn beta(&self) -> Option<f64> {
    self.beta
  }

  /// Generalised inverse of `∂_v C(· | v)`: the atom `v^{β/α}` for `y` in the jump `[(1-β) w, w]`, `w = v^{β(1-α)/α}`,
  /// closed form elsewhere; inside `[0, 1]` the `α = 1` and `β = 1` branches are never selected.
  pub(crate) fn inverse(alpha: f64, beta: f64) -> impl Fn(f64, f64) -> f64 {
    move |y, v| {
      let w = v.powf(beta * (1.0 - alpha) / alpha);
      if y == 0.0 {
        0.0
      } else if y < (1.0 - beta) * w {
        y * v.powf(beta) / (1.0 - beta)
      } else if y <= w {
        v.powf(beta / alpha)
      } else {
        y.powf(1.0 / (1.0 - alpha))
      }
    }
  }

  /// The pair behind a passed `check_fit`.
  fn resolve_params(&self) -> (f64, f64) {
    (self.alpha.expect("checked"), self.beta.expect("checked"))
  }

  /// `MarshallOlkin::inverse`, the generalised inverse of `∂_v C(· | v)`.
  fn unchecked_inverse(
    &self,
    y: ArrayView1<f64>,
    v: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    let (alpha, beta) = self.resolve_params();
    Ok(conditional_quantiles(y, v, Self::inverse(alpha, beta)))
  }
}

impl BivariateExt for MarshallOlkin {
  fn family(&self) -> BivariateFamily {
    BivariateFamily::MarshallOlkin
  }

  /// `α`, on the symmetric slice `α = β` only.
  fn theta(&self) -> Option<f64> {
    match (self.alpha, self.beta) {
      (Some(alpha), Some(beta)) if alpha == beta => Some(alpha),
      _ => None,
    }
  }

  /// Sets the symmetric slice `α = β = theta`; panics unless `validate_theta` admits it.
  fn set_theta(&mut self, theta: f64) {
    self.validate_theta(theta).unwrap_or_else(|e| panic!("{e}"));
    self.alpha = Some(theta);
    self.beta = Some(theta);
  }

  fn theta_bounds(&self) -> (f64, f64) {
    (0.0, 1.0)
  }

  fn invalid_thetas(&self) -> Vec<f64> {
    vec![0.0]
  }

  /// `NotFitted` before the pair is set, otherwise both shock rates in `(0, 1]`.
  fn check_fit(&self) -> Result<(), CopulaError> {
    let (Some(alpha), Some(beta)) = (self.alpha, self.beta) else {
      return Err(CopulaError::NotFitted);
    };
    for (name, value) in [("alpha", alpha), ("beta", beta)] {
      if !(value > 0.0 && value <= 1.0) {
        return Err(CopulaError::InvalidParameter {
          name,
          value,
          constraint: format!("0.0 < {name} <= 1.0"),
        });
      }
    }
    Ok(())
  }

  /// Absolutely continuous density: `0` exactly on the singular curve $u^\alpha = v^\beta$, and
  /// `(1 - \alpha) u^{-\alpha}` / `(1 - \beta) v^{-\beta}` in the two open sectors.
  fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let (alpha, beta) = self.resolve_params();
    on_open_square(x, |u, v| {
      let lhs = u.powf(alpha);
      let rhs = v.powf(beta);
      if (lhs - rhs).abs() < 1e-15 {
        // Singular curve: absolutely continuous density is 0; the singular
        // component carries the curve mass and is not returned here.
        0.0
      } else if lhs > rhs {
        (1.0 - alpha) * u.powf(-alpha)
      } else {
        (1.0 - beta) * v.powf(-beta)
      }
    })
  }

  /// CDF $C_{\alpha,\beta}(u,v) = \min(u^{1-\alpha} v, u v^{1-\beta})$.
  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let (alpha, beta) = self.resolve_params();
    copula_cdf(x, |u, v| {
      if u.powf(alpha) >= v.powf(beta) {
        u.powf(1.0 - alpha) * v
      } else {
        u * v.powf(1.0 - beta)
      }
    })
  }

  /// `∂_v C`: `u^{1-α}` where `u^α ≥ v^β`, `(1 - β) u v^{-β}` below; the jump
  /// `β v^{β(1-α)/α}` at `u = v^{β/α}` is the singular component's conditional mass.
  fn partial_derivative(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let (alpha, beta) = self.resolve_params();
    conditional_cdf(x, |u, v| {
      if u <= 0.0 {
        // At `α = 1` the law at `v = 0` is the point mass at 0, counted at 0.
        if v == 0.0 && alpha == 1.0 { 1.0 } else { 0.0 }
      } else if u >= 1.0 {
        1.0
      } else if u >= v.powf(beta / alpha) {
        u.powf(1.0 - alpha)
      } else {
        // `u·v^{−β}` first, so a subnormal `u` meets the factor `1 − β` only once the product is normal.
        (1.0 - beta) * (u * v.powf(-beta))
      }
    })
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

  /// The symmetric slice `α = β = θ`: `τ = θ/(2 − θ)`.
  fn tau_from_theta(&self, theta: f64) -> f64 {
    theta / (2.0 - theta)
  }

  fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError> {
    if !(tau > 0.0 && tau <= 1.0) {
      return Err(tau_out_of_range(tau, "0.0 < tau <= 1.0".into()));
    }
    admitted(self, tau, 2.0 * tau / (1.0 + tau))
  }

  /// `αβ/(α + β − αβ)`, the symmetric slice's `θ/(2 − θ)` at `α = β`.
  fn tau(&self) -> Option<f64> {
    match (self.alpha, self.beta) {
      (Some(a), Some(b)) if a == b => Some(self.tau_from_theta(a)),
      (Some(a), Some(b)) => Some(a * b / (a + b - a * b)),
      _ => None,
    }
  }

  /// $\lambda_U = \min(\alpha,\beta)$, carried entirely by the singular component on $u^\alpha = v^\beta$; no lower-tail dependence.
  /// Reference: Marshall, A.W., Olkin, I. (1967); Nelsen (2006) Example 5.22 (module header).
  fn tail_dependence(&self) -> Result<TailDependence<f64>, CopulaError> {
    self.check_fit()?;
    let (alpha, beta) = self.resolve_params();
    Ok(TailDependence {
      lower: 0.0,
      upper: alpha.min(beta),
    })
  }
}

#[cfg(test)]
mod tests;
