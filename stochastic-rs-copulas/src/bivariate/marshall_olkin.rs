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
//! Equivalently $C(u,v) = \min\!\big(u^{1-\alpha} v,\, u v^{1-\beta}\big)$.
//! Non-exchangeable in general; **has a singular component** on the curve
//! $u^\alpha = v^\beta$ carrying mass $\alpha\beta / (\alpha + \beta -
//! \alpha\beta)$. The remaining $1 - \alpha\beta/(\alpha+\beta-\alpha\beta)$
//! of probability mass is absolutely continuous on the two open sectors.
//!
//! Tail dependence: $\lambda_U = \min(\alpha, \beta)$ (upper),
//! $\lambda_L = 0$ (lower).
//!
//! Kendall's tau:
//! $\tau = \alpha\beta / (\alpha + \beta - \alpha\beta)$.
//!
//! [`BivariateExt::set_theta`] / [`BivariateExt::compute_theta`] operate on
//! the *symmetric* slice $\alpha = \beta = \theta$, in which case
//! $\tau = \theta / (2 - \theta) \iff \theta = 2\tau/(1 + \tau)$. The
//! asymmetric two-parameter case is supplied through
//! [`MarshallOlkin::with_alpha_beta`].
//!
//! Reference: Marshall, A.W., Olkin, I. (1967), "A multivariate exponential distribution", *JASA* 62(317), 30-44, DOI 10.1080/01621459.1967.10482885.
//! Reference: Nelsen, R.B. (2006), "An Introduction to Copulas", 2nd ed., Springer, §3.1.1, eq. 3.1.3; §2.9 (the conditional-distribution method with a quasi-inverse), DOI 10.1007/0-387-28678-0.

use ndarray::Array1;
use ndarray::Array2;

use crate::bivariate::CopulaType;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;

#[derive(Debug, Clone)]
pub struct MarshallOlkin {
  pub r#type: CopulaType,
  pub theta: Option<f64>,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
  /// Asymmetric Marshall-Olkin parameter $\alpha \in (0, 1]$. When
  /// `Some`, takes precedence over the symmetric `theta` field.
  pub alpha: Option<f64>,
  /// Asymmetric Marshall-Olkin parameter $\beta \in (0, 1]$.
  pub beta: Option<f64>,
}

impl Default for MarshallOlkin {
  fn default() -> Self {
    Self {
      r#type: CopulaType::MarshallOlkin,
      theta: None,
      tau: None,
      theta_bounds: (0.0, 1.0),
      invalid_thetas: vec![],
      alpha: None,
      beta: None,
    }
  }
}

impl MarshallOlkin {
  pub fn new() -> Self {
    Self::default()
  }

  /// Construct the asymmetric two-parameter variant $C_{\alpha,\beta}$.
  /// Both parameters must lie in $(0, 1]$.
  pub fn with_alpha_beta(alpha: f64, beta: f64) -> Self {
    assert!(
      alpha > 0.0 && alpha <= 1.0,
      "alpha must lie in (0, 1], got {alpha}"
    );
    assert!(
      beta > 0.0 && beta <= 1.0,
      "beta must lie in (0, 1], got {beta}"
    );
    Self {
      alpha: Some(alpha),
      beta: Some(beta),
      ..Self::default()
    }
  }

  /// Resolve the effective `(alpha, beta)` pair from either the asymmetric
  /// fields or the symmetric `theta` fallback.
  fn resolve_params(&self) -> (f64, f64) {
    match (self.alpha, self.beta) {
      (Some(a), Some(b)) => (a, b),
      _ => {
        let theta = self
          .theta
          .expect("Marshall-Olkin: neither (alpha, beta) nor theta is set");
        (theta, theta)
      }
    }
  }
}

impl BivariateExt for MarshallOlkin {
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
    // Reset the asymmetric slot so the symmetric value wins on resolve.
    self.alpha = None;
    self.beta = None;
  }

  /// Validates the parameterisation `resolve_params` uses: `alpha` and `beta` in (0, 1] when both
  /// are set, otherwise `theta` through `check_theta`.
  fn check_fit(&self) -> Result<(), CopulaError> {
    let (Some(alpha), Some(beta)) = (self.alpha, self.beta) else {
      return self.check_theta();
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

  /// Absolutely continuous density. Returns `0` exactly on the singular
  /// curve $u^\alpha = v^\beta$ and `(1 - \alpha) u^{-\alpha}` /
  /// `(1 - \beta) v^{-\beta}` in the two open sectors.
  fn pdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let (alpha, beta) = self.resolve_params();
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
      let lhs = u.powf(alpha);
      let rhs = v.powf(beta);
      if (lhs - rhs).abs() < 1e-15 {
        // Singular curve: absolutely continuous density is 0; the singular
        // component carries the curve mass and is not returned here.
        out[i] = 0.0;
      } else if lhs > rhs {
        out[i] = (1.0 - alpha) * u.powf(-alpha);
      } else {
        out[i] = (1.0 - beta) * v.powf(-beta);
      }
    }
    Ok(out)
  }

  /// CDF $C_{\alpha,\beta}(u,v) = \min(u^{1-\alpha} v, u v^{1-\beta})$.
  fn cdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let (alpha, beta) = self.resolve_params();
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
      let lhs = u.powf(alpha);
      let rhs = v.powf(beta);
      out[i] = if lhs >= rhs {
        u.powf(1.0 - alpha) * v
      } else {
        u * v.powf(1.0 - beta)
      };
    }
    Ok(out)
  }

  /// `∂_v C`: `u^{1-α}` where `u^α ≥ v^β`, `(1 - β) u v^{-β}` below, NaN for `v` outside `[0, 1]`; the jump
  /// `β v^{β(1-α)/α}` at `u = v^{β/α}` is the singular component's conditional mass.
  fn partial_derivative(&self, x: &Array2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let (alpha, beta) = self.resolve_params();
    let u_col = x.column(0);
    let v_col = x.column(1);
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i];
      if !(0.0..=1.0).contains(&v) {
        out[i] = f64::NAN;
        continue;
      }
      if u <= 0.0 {
        out[i] = 0.0;
        continue;
      }
      if u >= 1.0 {
        out[i] = 1.0;
        continue;
      }
      out[i] = if u >= v.powf(beta / alpha) {
        u.powf(1.0 - alpha)
      } else {
        (1.0 - beta) * u * v.powf(-beta)
      };
    }
    Ok(out)
  }

  /// Generalised inverse of `∂_v C(· | v)`: the atom `v^{β/α}` for `y` in the jump `[(1-β) w, w]`, `w = v^{β(1-α)/α}`,
  /// closed form elsewhere, NaN for `y` or `v` outside `[0, 1]`; the `α = 1` and `β = 1` branches never divide by zero.
  fn percent_point(&self, y: &Array1<f64>, v: &Array1<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let (alpha, beta) = self.resolve_params();
    Ok(
      y.iter()
        .zip(v.iter())
        .map(|(&y, &v)| {
          if !(0.0..=1.0).contains(&y) || !(0.0..=1.0).contains(&v) {
            return f64::NAN;
          }
          let w = v.powf(beta * (1.0 - alpha) / alpha);
          if y < (1.0 - beta) * w {
            y * v.powf(beta) / (1.0 - beta)
          } else if y <= w {
            v.powf(beta / alpha)
          } else {
            y.powf(1.0 / (1.0 - alpha))
          }
        })
        .collect(),
    )
  }

  /// Symmetric-slice Kendall's tau inversion: $\theta = 2\tau / (1 + \tau)$.
  /// For the asymmetric case, $\tau$ alone underdetermines $(\alpha, \beta)$ —
  /// supply both directly via [`MarshallOlkin::with_alpha_beta`].
  fn compute_theta(&self) -> f64 {
    let tau = self.tau.unwrap();
    if tau <= 0.0 {
      return 0.0_f64.max(f64::EPSILON);
    }
    if tau >= 1.0 - 1e-12 {
      return 1.0;
    }
    (2.0 * tau / (1.0 + tau)).clamp(0.0, 1.0)
  }

  /// Upper-tail dependence $\lambda_U = \min(\alpha,\beta)$, carried
  /// entirely by the singular component on $u^\alpha = v^\beta$; no
  /// lower-tail dependence.
  /// Reference: Marshall, A.W., Olkin, I. (1967); Nelsen (2006) Example
  /// 5.22 (module header).
  ///
  /// Cannot use [`BivariateExt::assert_theta_valid_for_tail_dependence`]
  /// directly: the legitimate [`MarshallOlkin::with_alpha_beta`]
  /// construction path leaves `theta` as `None`, which the generic
  /// `theta`-based guard would reject as unfit even though it's a valid
  /// configuration. Validates the *resolved* `(alpha, beta)` against
  /// their own domain instead — e.g. a raw `set_theta(5.0)` bypasses
  /// `with_alpha_beta`'s constructor asserts and would otherwise report a
  /// nonsensical `\lambda_U > 1`.
  fn tail_dependence(&self) -> TailDependence<f64> {
    let (alpha, beta) = self.resolve_params();
    if !(alpha > 0.0 && alpha <= 1.0) || !(beta > 0.0 && beta <= 1.0) {
      panic!(
        "tail_dependence requires a valid theta: alpha={alpha} and beta={beta} must each lie in (0, 1]"
      );
    }
    TailDependence {
      lower: 0.0,
      upper: alpha.min(beta),
    }
  }
}

#[cfg(test)]
mod tests;
