//! # Clayton
//!
//! $$
//! C_\theta(u,v)=\left(u^{-\theta}+v^{-\theta}-1\right)^{-1/\theta},\ \theta>0
//! $$
//!
use std::error::Error;
use std::f64;

use ndarray::Array1;
use ndarray::Array2;

use super::CopulaType;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;

#[derive(Debug, Clone)]
pub struct Clayton {
  pub r#type: CopulaType,
  pub theta: Option<f64>,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Default for Clayton {
  fn default() -> Self {
    Self {
      r#type: CopulaType::Clayton,
      theta: None,
      tau: None,
      theta_bounds: (0.0, f64::INFINITY),
      invalid_thetas: vec![],
    }
  }
}

impl Clayton {
  pub fn new() -> Self {
    Self::default()
  }
}

impl BivariateExt for Clayton {
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
    self.check_fit()?;

    let theta = self.theta.unwrap();
    Ok((1.0 / theta) * (t.powf(-theta) - 1.0))
  }

  /// Density $c(u,v) = (\theta+1)(uv)^{-\theta-1}\big(u^{-\theta}+v^{-\theta}-1\big)^{-(1+2\theta)/\theta}$
  /// — matches the mixed second partial of `cdf` for every `θ` in this
  /// family's valid range, checked directly. Now special-cased at the
  /// `θ = 0` independence boundary the same way [`Clayton::percent_point`]
  /// already is: naive substitution there hits a removable singularity
  /// (`b.powf(c)` with `b = 1` and `c = -∞`, an IEEE `pow` special case
  /// that evaluates to `1.0` regardless) that happens to leave `a` — not
  /// `1.0` — as the answer, i.e. `(uv)^{-1}`.
  fn pdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    let U = X.column(0);
    let V = X.column(1);

    let theta = self.theta.unwrap();

    if theta == 0.0 {
      return Ok(Array1::ones(U.len()));
    }

    let a = (theta + 1.0) * (&U * &V).powf(-theta - 1.0);
    let b = U.powf(-theta) + V.powf(-theta) - 1.0;
    let c = -(2.0 * theta + 1.0) / theta;
    Ok(a * b.powf(c))
  }

  /// At `θ=0` (independence), `C(u,v)=uv`; the general closed form
  /// `(u^{-θ}+v^{-θ}-1)^{-1/θ}` hits the same `1^{-∞}` removable
  /// singularity as `pdf` there and evaluates to the constant `1.0` for
  /// every `u,v > 0`, not `uv`.
  fn cdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    let U = X.column(0);
    let V = X.column(1);

    let V_all_zeros = V.iter().all(|&v| v == 0.0);
    let U_all_zeros = U.iter().all(|&u| u == 0.0);

    if V_all_zeros || U_all_zeros {
      let shape = V.shape();
      return Ok(Array1::zeros(shape[0]));
    }

    let theta = self.theta.unwrap();

    if theta == 0.0 {
      return Ok(&U * &V);
    }

    let mut cdfs = Array1::<f64>::zeros(U.len());

    for i in 0..U.len() {
      let u = U[i];
      let v = V[i];

      if u > 0.0 && v > 0.0 {
        cdfs[i] = (u.powf(-theta) + v.powf(-theta) - 1.0).powf(-1.0 / theta);
      } else {
        cdfs[i] = 0.0;
      }
    }

    Ok(cdfs)
  }

  /// Inverse conditional `u` solving `∂_v C(u,v) = p` (`p = y`): closed
  /// form `u = (1+v^{-θ}(p^{-θ/(1+θ)}-1))^{-1/θ}` for `θ ≠ 0`. At `θ=0`
  /// (independence, `C(u,v)=uv`), `∂_v C(u,v) = u`, so the inverse is the
  /// identity on the fresh uniform `p` — not `v`. The previous branch
  /// returned `V.clone()`, making every sampled pair exactly comonotonic
  /// (`U=V`, Kendall's τ ≈ 1) instead of independent — the same defect
  /// class as [`crate::bivariate::frank::Frank::percent_point`]'s
  /// pre-fix `θ = 0` branch.
  fn percent_point(&self, y: &Array1<f64>, V: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    let theta = self.theta.unwrap();

    if theta == 0.0 {
      return Ok(y.clone());
    }

    let a = y.powf(theta / (-1.0 - theta));
    let b = V.powf(theta);

    let b_all_zeros = b.iter().all(|&v| v == 0.0);

    if b_all_zeros {
      return Ok(Array1::ones(V.len()));
    }

    Ok(((a + &b - 1.0) / b).powf(-1.0 / theta))
  }

  /// At `θ=0` (independence, `C(u,v)=uv`), `∂_v C(u,v) = u`. The general
  /// formula hits the same `1^{-∞}` removable singularity as `pdf`/`cdf`
  /// there (`B = v^0+u^0-1 = 1`, raised to `(-1-θ)/θ → -∞`), leaving `A =
  /// v^{-1}` — not `u` — as the answer, so it needs its own branch.
  fn partial_derivative(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    let U = X.column(0);
    let V = X.column(1);

    let theta = self.theta.unwrap();

    if theta == 0.0 {
      return Ok(U.to_owned());
    }

    let A = V.powf(-theta - 1.0);

    if A.iter().all(|a| a.is_infinite()) {
      return Ok(Array1::zeros(V.len()));
    }

    let B = V.powf(-theta) + U.powf(-theta) - 1.0;
    let h = B.powf((-1.0 - theta) / theta);
    Ok(A * h)
  }

  fn compute_theta(&self) -> f64 {
    if self.tau.is_some() && self.tau.unwrap() == 1.0 {
      return f64::INFINITY;
    }

    let tau = self.tau.unwrap();

    2.0 * tau / (1.0 - tau)
  }

  /// Lower-tail dependence $\lambda_L = 2^{-1/\theta}$; Clayton has no
  /// upper-tail dependence ($\lambda_U = 0$).
  /// Reference: Nelsen, R.B. (2006), "An Introduction to Copulas", 2nd ed.,
  /// Springer, Example 5.21 / Table 5.1.
  fn tail_dependence(&self) -> TailDependence<f64> {
    self.assert_theta_valid_for_tail_dependence();
    let theta = self.theta.unwrap();
    TailDependence {
      lower: 2.0_f64.powf(-1.0 / theta),
      upper: 0.0,
    }
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  /// Clayton θ=2: λ_L = 2^{−1/2} = 0.707106781186…, λ_U = 0.
  #[test]
  fn clayton_tail_dependence_closed_form() {
    let mut c = Clayton::new();
    c.set_theta(2.0);
    let td = c.tail_dependence();
    assert!(
      (td.lower - 2.0_f64.powf(-0.5)).abs() < 1e-12,
      "λ_L={}, expected {}",
      td.lower,
      2.0_f64.powf(-0.5)
    );
    assert_eq!(td.upper, 0.0);
  }

  /// `sample` takes `&self`: no implementation mutates during sampling, so
  /// an immutable binding must be enough to draw from it.
  #[test]
  fn bivariate_sample_takes_shared_ref() {
    let c = Clayton::default();
    let _ = c.sample(8);
  }

  /// The `θ=0` regression test, mirroring
  /// `frank::tests::frank_independence_sample_kendall_tau_near_zero`
  /// exactly (same tolerance derivation — `6` standard errors of τ under
  /// the null of independence, Kendall & Gibbons (1990) — same 3-seed
  /// best-of). Before the `percent_point` fix, `θ=0` returned
  /// `V.clone()`, making `U=V` exactly for every sampled pair
  /// (comonotonic, τ≈1) instead of the independence `θ=0` is supposed to
  /// mean.
  #[test]
  fn clayton_independence_sample_kendall_tau_near_zero() {
    let n = 4000usize;
    let se = (2.0 * (2.0 * n as f64 + 5.0) / (9.0 * n as f64 * (n as f64 - 1.0))).sqrt();
    let tol = 6.0 * se;
    let c = Clayton {
      theta: Some(0.0),
      tau: Some(0.0),
      ..Clayton::default()
    };
    let best_abs_tau = [2718_u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let samples = c.sample_with_seed(n, seed).unwrap();
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

  /// `θ=0` is Clayton's own independence limit. `pdf`/`cdf`/
  /// `partial_derivative` had no branch for it at all — mirroring the
  /// gap `frank::tests::frank_cdf_at_independence_is_uv` documents for
  /// Frank's pre-fix `cdf` — so naive substitution hit a removable
  /// `1^{-∞}` singularity instead of resolving to the independence
  /// copula. Asserted exactly (not via a loose tolerance), matching this
  /// crate's convention for independence-limit checks.
  #[test]
  fn clayton_pdf_at_independence_is_one() {
    let c = Clayton {
      theta: Some(0.0),
      ..Clayton::default()
    };
    let pdf = c.pdf(&array![[0.2_f64, 0.9], [0.5, 0.5]]).unwrap();
    assert!(approx(pdf[0], 1.0, 1e-12), "got {}", pdf[0]);
    assert!(approx(pdf[1], 1.0, 1e-12), "got {}", pdf[1]);
  }

  #[test]
  fn clayton_cdf_at_independence_is_uv() {
    let c = Clayton {
      theta: Some(0.0),
      ..Clayton::default()
    };
    let cdf = c.cdf(&array![[0.3_f64, 0.7], [0.2, 0.9]]).unwrap();
    assert!(approx(cdf[0], 0.21, 1e-12), "got {}", cdf[0]);
    assert!(approx(cdf[1], 0.18, 1e-12), "got {}", cdf[1]);
  }

  /// `partial_derivative` computes $\partial_v C(u,v)$ (this family's own
  /// convention, stated in [`Clayton::percent_point`]'s doc). At
  /// independence, $C(u,v)=uv \Rightarrow \partial_v C = u$ — not the
  /// pre-fix `1^{-∞}` singularity's `v^{-1}` result.
  #[test]
  fn clayton_partial_derivative_at_independence_returns_u() {
    let c = Clayton {
      theta: Some(0.0),
      ..Clayton::default()
    };
    let pd = c
      .partial_derivative(&array![[0.3_f64, 0.6], [0.2, 0.9]])
      .unwrap();
    assert!(approx(pd[0], 0.3, 1e-12), "got {}", pd[0]);
    assert!(approx(pd[1], 0.2, 1e-12), "got {}", pd[1]);
  }

  /// `h(u | v) = v^{-θ-1} (u^{-θ} + v^{-θ} - 1)^{-(1+θ)/θ}` (Aas, Czado,
  /// Frigessi & Bakken 2009, Table 1 — the Clayton row) at θ = 2, evaluated
  /// with the same formula in Python:
  ///   th=2.0; u=0.3; v=0.6; v**(-th-1)*(u**(-th)+v**(-th)-1)**(-(1+th)/th)
  /// gives 0.10005136755229085, and 0.8004109404183268 with u and v swapped.
  /// Guards the degenerate all-infinite branch, which ndarray's inverted
  /// `is_all_infinite` used to select for every finite input.
  #[test]
  fn partial_derivative_matches_the_closed_form_h_function() {
    let mut c = Clayton::new();
    c.set_theta(2.0);
    let h = c
      .partial_derivative(&array![[0.3, 0.6], [0.6, 0.3]])
      .unwrap();
    assert!(
      (h[0] - 0.10005136755229085).abs() < 1e-14,
      "h(0.3 | 0.6) = {}",
      h[0]
    );
    assert!(
      (h[1] - 0.8004109404183268).abs() < 1e-14,
      "h(0.6 | 0.3) = {}",
      h[1]
    );
  }
}
