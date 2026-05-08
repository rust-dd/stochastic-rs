//! # Frank
//!
//! $$
//! C_\theta(u,v)=-\frac1\theta\log\!\left(1+\frac{(e^{-\theta u}-1)(e^{-\theta v}-1)}{e^{-\theta}-1}\right)
//! $$
//!
use core::f64;
use std::error::Error;

use gauss_quad::GaussLegendre;
use ndarray::Array1;
use ndarray::Array2;
use roots::SimpleConvergency;
use roots::find_root_brent;

use crate::bivariate::CopulaType;
use crate::traits::BivariateExt;

#[derive(Debug, Clone)]
pub struct Frank {
  pub r#type: CopulaType,
  pub theta: Option<f64>,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Frank {
  pub fn new(theta: Option<f64>, tau: Option<f64>) -> Self {
    Self {
      r#type: CopulaType::Frank,
      theta,
      tau,
      theta_bounds: (f64::NEG_INFINITY, f64::INFINITY),
      invalid_thetas: vec![0.0],
    }
  }
}

impl BivariateExt for Frank {
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

  fn generator(&self, t: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    let theta = self.theta.unwrap();
    let a = ((-theta * t).exp() - 1.0) / ((-theta).exp() - 1.0);
    let out = -(a.ln());
    Ok(out)
  }

  fn pdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    let U = X.column(0).to_owned();
    let V = X.column(1).to_owned();

    let theta = self.theta.unwrap();

    if theta == 0.0 {
      return Ok(Array1::ones(U.len()));
    }

    let num = (-theta * self._g(&Array1::ones(U.len()))?) * (1.0 + self._g(&(&U + &V))?);
    let aux = self._g(&U)? + self._g(&V)? + self._g(&Array1::ones(U.len()))?;
    let den = aux.pow2();
    Ok(num / den)
  }

  fn cdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    let U = X.column(0);
    let V = X.column(1);

    let theta = self.theta.unwrap();
    let num = ((-theta * &U).exp() - 1.0) * ((-theta * &V).exp() - 1.0);
    let den = (-theta).exp() - 1.0;
    let out = -1.0 / theta * (1.0 + num / den).ln();
    Ok(out)
  }

  #[allow(clippy::only_used_in_recursion)]
  fn percent_point(&self, y: &Array1<f64>, V: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    let theta = self.theta.unwrap();

    if theta == 0.0 {
      return Ok(V.clone());
    }

    let out = BivariateExt::percent_point(self, y, V)?;
    Ok(out)
  }

  fn partial_derivative(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    self.check_fit()?;

    let U = X.column(0).to_owned();
    let V = X.column(1).to_owned();

    let theta = self.theta.unwrap();

    if theta == 0.0 {
      return Ok(V.clone());
    }

    let num = self._g(&U)? * self._g(&V)? + self._g(&U)?;
    let den = self._g(&U)? + self._g(&V)? + self._g(&Array1::ones(U.len()))?;
    Ok(num / den)
  }

  fn compute_theta(&self) -> f64 {
    let tau = self.tau.unwrap();

    if tau.abs() < 1e-12 {
      return 0.0;
    }
    if tau >= 1.0 {
      return f64::INFINITY;
    }
    if tau <= -1.0 {
      return f64::NEG_INFINITY;
    }

    let residual = |theta: f64| Self::_tau_to_theta(tau, theta);
    let mut convergency = SimpleConvergency {
      eps: 1e-8,
      max_iter: 100,
    };
    let (lo, hi) = if tau > 0.0 {
      (1e-8_f64, 50.0_f64)
    } else {
      (-50.0_f64, -1e-8_f64)
    };
    find_root_brent(lo, hi, residual, &mut convergency).unwrap_or(0.0)
  }
}

impl Frank {
  fn _g(&self, z: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    Ok((-self.theta.unwrap() * z).exp() - 1.0)
  }

  /// Residual `τ(θ) − τ_target` for the Frank tau↔theta relation
  /// `τ(θ) = 1 − 4/θ + (4/θ²) · ∫₀^θ t/(eᵗ − 1) dt` (Genest-MacKay 1986).
  /// For θ < 0 the integral is taken in the negative direction
  /// (i.e. `−∫_θ^0`) so the formula is well-defined on `(−∞, ∞)\{0}`.
  fn _tau_to_theta(tau: f64, alpha: f64) -> f64 {
    if alpha.abs() < 1e-15 {
      // Independence limit: τ(0) = 0.
      return -tau;
    }
    let abs_a = alpha.abs();
    let integrand = |u: f64| {
      if u.abs() < 1e-15 {
        1.0
      } else {
        u / (u.exp() - 1.0)
      }
    };
    // The integrand t/(eᵗ−1) drops from 1 at t=0 to e^{-t} for large t.
    // Naive Gauss-Legendre over [0, |alpha|] under-resolves the spike near 0
    // for large |alpha|; split into chunks of width ≤ 1 with 8 nodes each
    // (effectively a piecewise high-order rule) for stable integration.
    let quad = GaussLegendre::new(std::num::NonZeroUsize::new(8).unwrap());
    let chunk_w = 1.0_f64;
    let n_chunks = (abs_a / chunk_w).ceil() as usize;
    let mut integral_pos = 0.0_f64;
    for k in 0..n_chunks {
      let lo = (k as f64) * chunk_w;
      let hi = ((k + 1) as f64 * chunk_w).min(abs_a);
      integral_pos += quad.integrate(lo, hi, integrand);
    }
    // For α < 0, ∫₀^α t/(eᵗ−1) dt = −α²/2 − ∫₀^|α| u/(eᵘ−1) du.
    // (Substitute u = −t and use u/(e⁻ᵘ−1) = −u − u/(eᵘ−1).)
    let integral = if alpha > 0.0 {
      integral_pos
    } else {
      -alpha * alpha / 2.0 - integral_pos
    };
    let tau_theta = 1.0 - 4.0 / alpha + 4.0 * integral / (alpha * alpha);
    tau_theta - tau
  }
}
