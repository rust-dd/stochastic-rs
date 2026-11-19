//! Two-step GMM of CIR on its exact transition moments, four conditions for three parameters.
//!
//! $$
//! dX_t = \theta(\mu - X_t)\,dt + \sigma\sqrt{X_t}\,dW_t,\qquad a = e^{-\theta\Delta},\qquad
//! g_t = \bigl(e_t,\ e_t X_t,\ u_t,\ u_t X_t\bigr)^\top
//! $$
//!
//! $$
//! e_t = X_{t+\Delta} - \mu - (X_t - \mu)\,a,\qquad J = T\,\bar g^\top \hat S^{-1} \bar g
//! $$
//!
//! $$
//! u_t = e_t^2 - X_t\,\frac{\sigma^2}{\theta}(a - a^2) - \mu\,\frac{\sigma^2}{2\theta}(1 - a)^2
//! $$
//!
//! Stage 1 weighs by the identity, on the path over its mean, stage 2 by `Ŝ⁻¹`; `J ~ χ²₁`.
//!
//! Reference: Hansen (1982), Econometrica 50(4), 1029-1054, DOI: 10.2307/1912775
//! Reference: Cox, Ingersoll & Ross (1985), Econometrica 53(2), 385-407, DOI: 10.2307/1911242
//! Reference: Chan et al. (1992), J. Fin. 47(3), 1209-1227, DOI: 10.1111/j.1540-6261.1992.tb04011.x

use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::special::erfc;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::params::VarianceBoundary;

use crate::StatsError;
use crate::optim::nelder_mead;
use crate::qmle::Ar1;
use crate::qmle::Carrier;
use crate::qmle::DiffusionKind;
use crate::qmle::Transition;
use crate::qmle::ar1_start;
use crate::qmle::assert_dt;
use crate::qmle::diffusion_path;
use crate::qmle::in_data_units;
use crate::stationarity::common::unit_scaled;
use crate::traits::FloatExt;

/// The two-step GMM fit of the CIR process, in [`Cir`]'s names: `theta` the speed, `mu` the level.
#[derive(Debug, Clone)]
pub struct GmmCirFit {
  pub theta: f64,
  pub mu: f64,
  pub sigma: f64,
  /// Hansen's over-identification statistic, `χ²₁` under the model.
  pub j_stat: f64,
  pub j_pvalue: f64,
  /// Simplex iterations over both stages.
  pub iterations: usize,
  /// Whether both stages converged.
  pub converged: bool,
}

impl GmmCirFit {
  /// The fitted [`Cir`]; panics as [`Cir::new`].
  pub fn to_cir<S: SeedExt>(
    &self,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    boundary: VarianceBoundary,
    seed: S,
  ) -> Cir<f64, S> {
    Cir::new(self.theta, self.mu, self.sigma, n, x0, t, boundary, seed)
  }
}

/// Two-step GMM of CIR every `dt`; `Err` below 5 points, a negative or non-finite one, a constant
/// or exact-fit path, singular `Ŝ`, non-finite start, `Ŝ`, `J` or fit; panics unless `0 < dt < ∞`.
pub fn fit_cir_gmm<T: FloatExt>(series: ArrayView1<T>, dt: f64) -> Result<GmmCirFit, StatsError> {
  assert_dt(dt);
  // `Ŝ` sums one outer product per transition, so its four moments need four transitions. At unit
  // scale no moment leaves the double range, whatever the path's unit.
  let (x, unit) = unit_scaled(diffusion_path(series, DiffusionKind::Cir, 5)?);
  // On an exact fit `σ̂` is where the simplex stops on its way to zero, which moves with the scale.
  Ar1::of(&x, DiffusionKind::Cir)?;
  let nobs = x.len();

  // The `T − 1` moment rows `g_t` at `(θ, μ, σ)` and their column mean.
  let moments = |theta: f64, mu: f64, sigma: f64| -> (Vec<[f64; 4]>, [f64; 4]) {
    let transition = Transition::new(DiffusionKind::Cir, theta, mu, sigma, dt);
    let mut rows = Vec::with_capacity(nobs - 1);
    let mut mean = [0.0_f64; 4];
    for t in 0..nobs - 1 {
      let xt = x[t];
      let (m, var) = transition.moments(xt);
      let e = x[t + 1] - m;
      let u = e * e - var;
      let row = [e, e * xt, u, u * xt];
      for k in 0..4 {
        mean[k] += row[k];
      }
      rows.push(row);
    }
    let inv_t = 1.0 / (nobs - 1) as f64;
    for entry in &mut mean {
      *entry *= inv_t;
    }
    (rows, mean)
  };

  let objective = |theta: f64, mu: f64, sigma: f64, w: &[[f64; 4]; 4]| -> f64 {
    let (_, g) = moments(theta, mu, sigma);
    let mut q = 0.0;
    for i in 0..4 {
      for j in 0..4 {
        q += g[i] * w[i][j] * g[j];
      }
    }
    q
  };

  let carrier = Carrier::new(&x, DiffusionKind::Cir)?;
  let start = ar1_start(&x, dt, DiffusionKind::Cir, carrier)?;

  // The identity on the moments of the path over its mean, of degrees 1, 2, 2 and 3 in the path:
  // on the path's own it would weigh them by the data's unit.
  let stage1 = {
    let mut w = [[0.0; 4]; 4];
    for (i, degree) in [1, 2, 2, 3].into_iter().enumerate() {
      w[i][i] = carrier.mean().powi(-2 * degree);
    }
    w
  };

  let (p1, it1, conv1) = nelder_mead(start, 2000, |p| {
    objective(p[0].exp(), p[1].exp(), p[2].exp(), &stage1)
  });
  let (theta1, mu1, s1) = (p1[0].exp(), p1[1].exp(), p1[2].exp());

  // Stage 2 weights by `Ŝ⁻¹` at the stage-1 estimate, Hansen's efficient choice.
  let (rows1, _) = moments(theta1, mu1, s1);
  let mut s_hat = [[0.0_f64; 4]; 4];
  for row in &rows1 {
    for i in 0..4 {
      for j in 0..4 {
        s_hat[i][j] += row[i] * row[j];
      }
    }
  }
  let inv_t = 1.0 / rows1.len() as f64;
  for entry in s_hat.iter_mut().flatten() {
    *entry *= inv_t;
  }
  let w2 = weighting_matrix(&s_hat)?;

  let (p2, it2, conv2) = nelder_mead([p1[0], p1[1], p1[2]], 2000, |p| {
    objective(p[0].exp(), p[1].exp(), p[2].exp(), &w2)
  });
  let (theta, mu, sigma) = (p2[0].exp(), p2[1].exp(), p2[2].exp());

  let (j_stat, j_pvalue) = j_test(rows1.len(), objective(theta, mu, sigma, &w2))?;
  let [theta, mu, sigma] = in_data_units(DiffusionKind::Cir, [theta, mu, sigma], unit)?;

  Ok(GmmCirFit {
    theta,
    mu,
    sigma,
    j_stat,
    j_pvalue,
    iterations: it1 + it2,
    converged: conv1 && conv2,
  })
}

/// `Ŝ⁻¹` through `C = D^{-1/2} Ŝ D^{-1/2}`, `D = diag Ŝ`, whose pivots no unit moves; `Numerical`
/// for a non-finite `Ŝ`, `Singular` for a zero diagonal or a pivot of `C` below `1e-12`.
fn weighting_matrix(s_hat: &[[f64; 4]; 4]) -> Result<[[f64; 4]; 4], StatsError> {
  if s_hat.iter().flatten().any(|v| !v.is_finite()) {
    return Err(StatsError::Numerical {
      stage: "GMM weighting matrix",
    });
  }
  let singular = StatsError::Singular("GMM weighting matrix");
  let mut inv_sd = [0.0; 4];
  for (i, row) in s_hat.iter().enumerate() {
    if row[i] == 0.0 {
      return Err(singular);
    }
    inv_sd[i] = 1.0 / row[i].sqrt();
  }
  let mut c = [[0.0; 4]; 4];
  for i in 0..4 {
    for j in 0..4 {
      c[i][j] = inv_sd[i] * s_hat[i][j] * inv_sd[j];
    }
  }
  let mut w = invert4(&c).ok_or(singular)?;
  for i in 0..4 {
    for j in 0..4 {
      w[i][j] *= inv_sd[i] * inv_sd[j];
    }
  }
  Ok(w)
}

/// Gauss–Jordan inverse with partial pivoting; `None` at a pivot that is NaN or below `1e-12`, the
/// rounding of a unit-diagonal matrix that is singular.
fn invert4(m: &[[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
  let mut a = *m;
  let mut inv = {
    let mut e = [[0.0; 4]; 4];
    for (i, row) in e.iter_mut().enumerate() {
      row[i] = 1.0;
    }
    e
  };
  for col in 0..4 {
    // Partial pivot.
    let mut pivot = col;
    for r in (col + 1)..4 {
      if a[r][col].abs() > a[pivot][col].abs() {
        pivot = r;
      }
    }
    let p = a[pivot][col];
    if p.is_nan() || p.abs() < 1e-12 {
      return None;
    }
    a.swap(col, pivot);
    inv.swap(col, pivot);
    let d = a[col][col];
    for k in 0..4 {
      a[col][k] /= d;
      inv[col][k] /= d;
    }
    for r in 0..4 {
      if r != col {
        let factor = a[r][col];
        for k in 0..4 {
          a[r][k] -= factor * a[col][k];
          inv[r][k] -= factor * inv[col][k];
        }
      }
    }
  }
  Some(inv)
}

/// Hansen's `J = T q` on `T` transitions and its `χ²₁` tail `erfc(√(J/2))`; `Numerical` when `J`,
/// and so the stage-2 objective `q`, is not finite.
fn j_test(t: usize, q: f64) -> Result<(f64, f64), StatsError> {
  let j = t as f64 * q;
  if !j.is_finite() {
    return Err(StatsError::Numerical {
      stage: "J statistic",
    });
  }
  // A quadratic form in a positive-definite `Ŝ⁻¹`: a negative `J` is its rounding.
  Ok((j, erfc((j.max(0.0) / 2.0).sqrt())))
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_stochastic::traits::ProcessExt;

  use super::*;
  use crate::qmle::test_support::simulate_cir;

  /// Best of three pinned paths: the level's band sits near one standard error, so one path alone
  /// is a coin flip.
  #[test]
  fn gmm_cir_recovers_params_from_exact_path() {
    let (tht, mt, st) = (2.0, 0.04, 0.25);
    let dt = 1.0 / 252.0;
    let fits = [7u64, 13, 29].map(|seed| {
      let path = simulate_cir(tht, mt, st, 0.04, dt, 8000, seed);
      (seed, fit_cir_gmm(path.view(), dt).unwrap())
    });
    let ok = fits.iter().any(|(_, r)| {
      r.converged
        && (r.mu - mt).abs() / mt < 0.12
        && (r.theta - tht).abs() / tht < 0.5
        && (r.sigma - st).abs() / st < 0.3
    });
    let report = fits
      .iter()
      .map(|(s, r)| format!("seed {s}: θ={:.4} μ={:.4} σ={:.4}", r.theta, r.mu, r.sigma))
      .collect::<Vec<_>>()
      .join("; ");
    assert!(ok, "no seed recovered (θ={tht}, μ={mt}, σ={st}) — {report}");
  }

  /// The fit's `theta` is the speed and `mu` the level, as in `Cir`, so the conversion hands them
  /// over.
  #[test]
  fn a_gmm_fit_round_trips_into_the_cir_it_names() {
    let truth = Cir::<f64, _>::new(
      5.0,
      0.04,
      0.2,
      10_000,
      Some(0.04),
      Some(100.0),
      VarianceBoundary::Reflect,
      Deterministic::new(5),
    );
    let path = truth.sample();
    let fit = fit_cir_gmm(path.view(), 100.0 / 9_999.0).unwrap();
    assert!((fit.theta - 5.0).abs() < 1.5, "speed {}", fit.theta);
    assert!((fit.mu - 0.04).abs() < 0.01, "level {}", fit.mu);
    let rebuilt = fit.to_cir(
      10_000,
      Some(0.04),
      Some(100.0),
      VarianceBoundary::Reflect,
      Deterministic::new(6),
    );
    assert_eq!(
      (rebuilt.theta, rebuilt.mu, rebuilt.sigma),
      (fit.theta, fit.mu, fit.sigma)
    );
    assert_eq!(rebuilt.sample().len(), 10_000);
  }

  /// `J` is `χ²₁` under the model, so a correct model is rejected at 1 % on one path in a hundred:
  /// the best of three pinned paths does not reject it.
  #[test]
  fn gmm_cir_j_test_does_not_reject_correct_model() {
    let dt = 1.0 / 252.0;
    let pvalues = [11u64, 12, 13].map(|seed| {
      let path = simulate_cir(1.5, 0.05, 0.3, 0.05, dt, 6000, seed);
      fit_cir_gmm(path.view(), dt).unwrap().j_pvalue
    });
    assert!(
      pvalues.iter().any(|&p| p > 0.01),
      "correctly-specified CIR rejected at 1 % on every path, J p-values {pvalues:?}"
    );
  }

  /// A path of zeros with one spike has three distinct moment rows, so `Ŝ` has rank 3 at most.
  #[test]
  fn a_singular_weighting_matrix_is_an_err() {
    let s = Array1::from_iter((0..100).map(|t| if t == 50 { 1.0 } else { 0.0 }));
    assert_eq!(
      fit_cir_gmm(s.view(), 1.0 / 252.0).unwrap_err(),
      StatsError::Singular("GMM weighting matrix")
    );
  }

  /// Four transitions are the fewest that give `Ŝ` full rank, and its equilibrated pivots see it.
  #[test]
  fn five_points_are_enough_for_the_weighting_matrix() {
    let path = simulate_cir(2.0, 0.04, 0.25, 0.04, 1.0 / 252.0, 5, 3);
    assert!(fit_cir_gmm(path.view(), 1.0 / 252.0).is_ok());
  }

  #[test]
  fn a_non_finite_weighting_matrix_is_a_numerical_error() {
    let mut s_hat = [[1.0; 4]; 4];
    s_hat[1][2] = f64::INFINITY;
    assert_eq!(
      weighting_matrix(&s_hat).unwrap_err(),
      StatsError::Numerical {
        stage: "GMM weighting matrix"
      }
    );
  }

  #[test]
  fn invert4_fails_on_a_nan_pivot() {
    assert_eq!(invert4(&[[f64::NAN; 4]; 4]), None);
    let mut m = [[0.0; 4]; 4];
    for (i, row) in m.iter_mut().enumerate() {
      row[i] = 1.0;
    }
    m[2][2] = 1e-13;
    assert_eq!(invert4(&m), None);
  }

  /// `J` and its `χ²₁` tail `erfc(√(J/2))`, which keeps its digits where `1 − erf` is `0`.
  #[test]
  fn the_j_test_answers_its_chi_square_tail_and_refuses_a_non_finite_objective() {
    let (j, p) = j_test(100, 1.0).unwrap();
    assert_eq!(j, 100.0);
    assert!((p / 1.523970604832105e-23 - 1.0).abs() < 1e-12, "p = {p:e}");
    assert_eq!(j_test(100, -1e-18).unwrap().1, 1.0);
    for q in [f64::NAN, f64::INFINITY] {
      assert_eq!(
        j_test(100, q).unwrap_err(),
        StatsError::Numerical {
          stage: "J statistic"
        }
      );
    }
  }

  /// Four points are three transitions, one short of a full-rank `Ŝ`.
  #[test]
  fn gmm_cir_rejects_a_short_series() {
    let s = Array1::from(vec![0.04, 0.05, 0.045, 0.05]);
    assert_eq!(
      fit_cir_gmm(s.view(), 1.0 / 252.0).unwrap_err(),
      StatsError::InsufficientData { needed: 5, got: 4 }
    );
  }

  #[test]
  fn gmm_cir_rejects_a_negative_observation() {
    let s = Array1::from(vec![0.04, 0.05, 0.045, -0.01, 0.04, 0.05]);
    assert_eq!(
      fit_cir_gmm(s.view(), 1.0 / 252.0).unwrap_err(),
      StatsError::InvalidObservation {
        index: 3,
        value: -0.01,
        constraint: "finite and >= 0"
      }
    );
  }

  #[test]
  #[should_panic(expected = "dt must satisfy `dt.is_finite() && dt > 0.0`, got dt = 0.0")]
  fn gmm_cir_panics_on_bad_dt() {
    let s = Array1::from(vec![0.04, 0.05, 0.045, 0.05]);
    let _ = fit_cir_gmm(s.view(), 0.0);
  }
}
