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
use crate::traits::TailDependence;

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
      invalid_thetas: vec![],
    }
  }
}

/// Unfit placeholder (`theta = tau = None`) — matches the zero-arg
/// `new()`-equivalent shape of this crate's other 11 bivariate families
/// (e.g. `Clayton::default()`); delegates to [`Frank::new`] so it can never
/// drift from what that constructor already produces.
impl Default for Frank {
  fn default() -> Self {
    Self::new(None, None)
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

    if theta == 0.0 {
      return Ok(&U * &V);
    }

    let num = ((-theta * &U).exp() - 1.0) * ((-theta * &V).exp() - 1.0);
    let den = (-theta).exp() - 1.0;
    let out = -1.0 / theta * (1.0 + num / den).ln();
    Ok(out)
  }

  fn percent_point(&self, y: &Array1<f64>, V: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    let theta = self.theta.unwrap();

    if theta == 0.0 {
      return Ok(y.clone());
    }

    self.percent_point_numerical(y, V)
  }

  fn partial_derivative(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    self.check_fit()?;

    let U = X.column(0).to_owned();
    let V = X.column(1).to_owned();

    let theta = self.theta.unwrap();

    if theta == 0.0 {
      return Ok(U.clone());
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

  /// Frank has no tail dependence in either tail, for any $\theta$.
  /// Reference: Nelsen, R.B. (2006), "An Introduction to Copulas", 2nd ed.,
  /// Springer, Table 5.1.
  fn tail_dependence(&self) -> TailDependence<f64> {
    self.assert_theta_valid_for_tail_dependence();
    TailDependence {
      lower: 0.0,
      upper: 0.0,
    }
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

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  #[test]
  fn frank_tail_dependence_is_zero() {
    let f = Frank::new(Some(6.0), None);
    let td = f.tail_dependence();
    assert_eq!(td.lower, 0.0);
    assert_eq!(td.upper, 0.0);
  }

  /// θ=0 is Frank's independence limit (`invalid_thetas` no longer rejects
  /// it) and must actually behave like one: `pdf(u,v) = 1` everywhere.
  #[test]
  fn frank_pdf_at_independence_is_one() {
    let f = Frank::new(Some(0.0), None);
    let pdf = f.pdf(&array![[0.2_f64, 0.9], [0.5, 0.5]]).unwrap();
    assert!(approx(pdf[0], 1.0, 1e-12), "got {}", pdf[0]);
    assert!(approx(pdf[1], 1.0, 1e-12), "got {}", pdf[1]);
  }

  /// `cdf` had no `theta == 0.0` branch at all: unblocking the boundary
  /// without one would leave `C(u,v) = -1/theta * ln(1 + 0/0)` (both the
  /// generator ratio's numerator and denominator vanish identically at
  /// theta=0) evaluating to `NaN` for every input. The independence copula
  /// is `C(u,v) = uv`.
  #[test]
  fn frank_cdf_at_independence_is_uv() {
    let f = Frank::new(Some(0.0), None);
    let cdf = f.cdf(&array![[0.3_f64, 0.7], [0.2, 0.9]]).unwrap();
    assert!(approx(cdf[0], 0.21, 1e-12), "got {}", cdf[0]);
    assert!(approx(cdf[1], 0.18, 1e-12), "got {}", cdf[1]);
  }

  /// `partial_derivative` computes $\partial_v C(u,v)$ (the same "second
  /// argument, fixed conditioning value" convention as
  /// [`crate::bivariate::gaussian::GaussianCopula::partial_derivative`]).
  /// At independence, $C(u,v)=uv \Rightarrow \partial_v C = u$ — not `v`.
  #[test]
  fn frank_partial_derivative_at_independence_returns_u() {
    let f = Frank::new(Some(0.0), None);
    let pd = f
      .partial_derivative(&array![[0.3_f64, 0.6], [0.2, 0.9]])
      .unwrap();
    assert!(approx(pd[0], 0.3, 1e-12), "got {}", pd[0]);
    assert!(approx(pd[1], 0.2, 1e-12), "got {}", pd[1]);
  }

  /// The test that decides the θ=0 boundary question: at Frank's
  /// independence limit, `percent_point`'s conditional inverse must not
  /// depend on the conditioning variable `V` at all, so sampled pairs
  /// must be empirically independent — Kendall's τ statistically
  /// indistinguishable from 0.
  ///
  /// Tolerance is `6` standard errors of τ under the null of independence
  /// (Kendall & Gibbons (1990), *Rank Correlation Methods*: `Var(τ) =
  /// 2(2n+5) / (9n(n-1))` for large `n`), derived from the sample size
  /// rather than tuned; three pinned seeds, best-of, per this crate's
  /// statistical-assertion convention.
  ///
  /// This must fail (and, before the `percent_point` fix, did fail) against
  /// the pre-fix `if theta == 0.0 { return Ok(V.clone()) }`: returning the
  /// conditioning variable instead of the fresh uniform `y` makes `U = V`
  /// exactly for every sampled pair — perfectly comonotonic, τ ≈ 1 — the
  /// exact opposite of the independence the caller asked for.
  #[test]
  fn frank_independence_sample_kendall_tau_near_zero() {
    let n = 4000usize;
    let se = (2.0 * (2.0 * n as f64 + 5.0) / (9.0 * n as f64 * (n as f64 - 1.0))).sqrt();
    let tol = 6.0 * se;
    let f = Frank::new(Some(0.0), Some(0.0));
    let best_abs_tau = [2718_u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let samples = f.sample_with_seed(n, seed).unwrap();
        let u = samples.column(0).to_vec();
        let v = samples.column(1).to_vec();
        let (tau, ..) =
          kendalls::tau_b_with_comparator(&u, &v, |a: &f64, b: &f64| a.partial_cmp(b).unwrap())
            .unwrap();
        tau.abs()
      })
      .fold(f64::INFINITY, f64::min);
    assert!(
      best_abs_tau < tol,
      "best |tau| across 3 seeds = {best_abs_tau}, expected < {tol} (6 SE, SE={se}, n={n})"
    );
  }
}
