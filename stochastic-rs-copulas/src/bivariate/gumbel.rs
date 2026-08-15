//! # Gumbel
//!
//! $$
//! C_\theta(u,v)=\exp\!\left(-\left(({-\ln u})^\theta+({-\ln v})^\theta\right)^{1/\theta}\right),\ \theta\ge1
//! $$
//!
use std::error::Error;

use ndarray::Array1;
use ndarray::Array2;

use super::CopulaType;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;

#[derive(Debug, Clone)]
pub struct Gumbel {
  pub r#type: CopulaType,
  pub theta: Option<f64>,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Gumbel {
  pub fn new(theta: Option<f64>, tau: Option<f64>) -> Self {
    Self {
      r#type: CopulaType::Gumbel,
      theta,
      tau,
      theta_bounds: (1.0, f64::INFINITY),
      invalid_thetas: vec![0.0],
    }
  }
}

/// Unfit placeholder (`theta = tau = None`) — matches the zero-arg
/// `new()`-equivalent shape of this crate's other 11 bivariate families
/// (e.g. `Clayton::default()`); delegates to [`Gumbel::new`] so it can never
/// drift from what that constructor already produces.
impl Default for Gumbel {
  fn default() -> Self {
    Self::new(None, None)
  }
}

impl BivariateExt for Gumbel {
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
    Ok((-t.ln()).powf(self.theta.unwrap()))
  }

  fn pdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    let U = X.column(0);
    let V = X.column(1);

    let theta = self.theta.unwrap();

    if theta == 1.0 {
      return Ok(Array1::ones(U.len()));
    }

    let a = (&U * &V).powf(-1.0);
    let tmp = (-U.ln()).powf(theta) + (-V.ln()).powf(theta);
    let b = tmp.powf(-2.0 + 2.0 / theta);
    let c = (U.ln() * V.ln()).powf(theta - 1.0);
    let d = 1.0 + (theta - 1.0) * tmp.powf(-1.0 / theta);
    let out = self.cdf(X)? * a * b * c * d;

    Ok(out)
  }

  fn cdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    let U = X.column(0);
    let V = X.column(1);

    let theta = self.theta.unwrap();

    if theta == 1.0 {
      return Ok(&U * &V);
    }

    let h = (-U.ln()).powf(theta) + (-V.ln()).powf(theta);
    let h = -h.powf(1.0 / theta);
    let cdfs = h.exp();

    Ok(cdfs)
  }

  fn percent_point(&self, y: &Array1<f64>, V: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    if self.theta.unwrap() == 1.0 {
      return Ok(y.to_owned());
    }

    self.percent_point_numerical(y, V)
  }

  /// $\partial_v C(u,v) = C(u,v)\big((-\ln u)^\theta+(-\ln
  /// v)^\theta\big)^{1/\theta-1}(-\ln v)^{\theta-1}/v$. At `θ=1`
  /// (independence, `C(u,v)=uv`), `∂_v C(u,v) = u` — not `v`. The
  /// previous branch returned `V.to_owned()`, the same defect class as
  /// [`crate::bivariate::frank::Frank::partial_derivative`]'s pre-fix
  /// `θ = 0` branch. Note that [`Gumbel::percent_point`]'s own `θ = 1`
  /// branch was already correct (it returns the fresh uniform directly,
  /// bypassing this method entirely), so `Gumbel::sample` was never
  /// affected by this bug — only a direct `partial_derivative` call at
  /// `θ = 1` was wrong.
  fn partial_derivative(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    self.check_fit()?;

    let U = X.column(0);
    let V = X.column(1);

    let theta = self.theta.unwrap();

    if theta == 1.0 {
      return Ok(U.to_owned());
    }

    let t1 = (-U.ln()).powf(theta);
    let t2 = (-V.ln()).powf(theta);
    let p1 = self.cdf(X)?;
    let p2 = (t1 + t2).powf(-1.0 + 1.0 / theta);
    let p3 = (-V.ln()).powf(theta - 1.0);
    let out = p1 * p2 * p3 / V;

    Ok(out)
  }

  fn compute_theta(&self) -> f64 {
    let tau = self.tau.unwrap();

    if tau >= 1.0 {
      return f64::INFINITY;
    }

    1.0 / (1.0 - tau)
  }

  /// Upper-tail dependence $\lambda_U = 2 - 2^{1/\theta}$; Gumbel has no
  /// lower-tail dependence ($\lambda_L = 0$).
  /// Reference: Nelsen, R.B. (2006), "An Introduction to Copulas", 2nd ed.,
  /// Springer, Table 5.1.
  fn tail_dependence(&self) -> TailDependence<f64> {
    self.assert_theta_valid_for_tail_dependence();
    let theta = self.theta.unwrap();
    TailDependence {
      lower: 0.0,
      upper: 2.0 - 2.0_f64.powf(1.0 / theta),
    }
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  /// Direct regression pin for the `θ=1` `partial_derivative` bug: at
  /// independence (`C(u,v)=uv ⟹ ∂_v C = u`), mirrors
  /// `frank::tests::frank_partial_derivative_at_independence_returns_u`.
  /// This is the test that actually exercises the bug fixed above — see
  /// the next test's doc for why a Kendall's-τ sampling check does not.
  #[test]
  fn gumbel_partial_derivative_at_independence_returns_u() {
    let g = Gumbel::new(Some(1.0), None);
    let pd = g
      .partial_derivative(&array![[0.3_f64, 0.6], [0.2, 0.9]])
      .unwrap();
    assert!((pd[0] - 0.3).abs() < 1e-12, "got {}", pd[0]);
    assert!((pd[1] - 0.2).abs() < 1e-12, "got {}", pd[1]);
  }

  /// Kendall's-τ sampling check at `θ=1`, same construction as
  /// `clayton::tests::clayton_independence_sample_kendall_tau_near_zero`
  /// and `frank::tests::frank_independence_sample_kendall_tau_near_zero`.
  /// Unlike those two, this one would pass even without the
  /// `partial_derivative` fix above: `Gumbel::percent_point` already had
  /// its own correct `θ=1` branch, so `Gumbel::sample` never routed
  /// through the buggy `partial_derivative` at this boundary. Kept
  /// anyway as a direct regression guard on `sample`/`percent_point`
  /// itself, matching this crate's standing convention of pinning
  /// independence-limit sampling behavior for every family that has one.
  #[test]
  fn gumbel_independence_sample_kendall_tau_near_zero() {
    let n = 4000usize;
    let se = (2.0 * (2.0 * n as f64 + 5.0) / (9.0 * n as f64 * (n as f64 - 1.0))).sqrt();
    let tol = 6.0 * se;
    let g = Gumbel::new(Some(1.0), Some(0.0));
    let best_abs_tau = [2718_u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let samples = g.sample_with_seed(n, seed).unwrap();
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

  /// Gumbel θ=2: λ_U = 2 − √2 = 0.585786437626905.
  #[test]
  fn gumbel_tail_dependence_closed_form() {
    let g = Gumbel::new(Some(2.0), None);
    let td = g.tail_dependence();
    assert!(
      (td.upper - (2.0 - std::f64::consts::SQRT_2)).abs() < 1e-12,
      "λ_U={}, expected {}",
      td.upper,
      2.0 - std::f64::consts::SQRT_2
    );
    assert_eq!(td.lower, 0.0);
  }

  /// θ=0.5 < `theta_bounds().0 = 1.0` — e.g. reachable via `fit()` on data
  /// with slightly negative empirical τ, since `_compute_theta` discards
  /// its own `check_theta()` result. Must panic, not silently return a
  /// nonsensical (negative) `λ_U`.
  #[test]
  #[should_panic(expected = "tail_dependence requires a valid theta")]
  fn gumbel_tail_dependence_panics_on_invalid_theta() {
    let g = Gumbel::new(Some(0.5), None);
    let _ = g.tail_dependence();
  }
}
