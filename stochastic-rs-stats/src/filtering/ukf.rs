//! Unscented Kalman filter, one forward step.
//!
//! `2n + 1` sigma points go through the transition and the measurement; the step is their moments.
//!
//! Reference: Julier & Uhlmann (2004), Proc. IEEE 92(3), 401-422, DOI: 10.1109/JPROC.2003.823141
//! Reference: Wan & van der Merwe (2000), IEEE AS-SPCC, 153-158, DOI: 10.1109/ASSPCC.2000.882463

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use stochastic_rs_distributions::linalg::inverse;
use stochastic_rs_distributions::linalg::spd_cholesky_lower;

use crate::StatsError;

/// State estimate (mean + covariance) on entry to / exit from a UKF step.
#[derive(Debug, Clone)]
pub struct UkfState {
  /// State posterior mean.
  pub mean: Array1<f64>,
  /// State posterior covariance.
  pub covariance: Array2<f64>,
}

/// Sigma-point parameters of [`unscented_kalman_step`] (Wan & van der Merwe 2000); all finite,
/// `alpha > 0`, `n + kappa > 0` (`n` the state dimension) and `n + lambda` finite and positive.
#[derive(Debug, Clone, Copy)]
pub struct UkfTuning {
  /// Spread of the sigma points; `lambda = alpha^2 (n + kappa) - n` cancels `n + lambda` to 0
  /// below about 1e-8 and overflows it above about 1e154.
  pub alpha: f64,
  /// Prior knowledge of the state's distribution; 2 is optimal for a Gaussian.
  pub beta: f64,
  /// Secondary scaling of the sigma points.
  pub kappa: f64,
}

impl Default for UkfTuning {
  fn default() -> Self {
    Self {
      alpha: 1e-3,
      beta: 2.0,
      kappa: 0.0,
    }
  }
}

/// One UKF predict-update of `prior` with sigma points set by `tuning`; `Err` on a non-PD prior or
/// a singular innovation covariance; panics unless the shapes agree and `tuning` is in its range.
pub fn unscented_kalman_step<F, H>(
  prior: &UkfState,
  transition: F,
  measurement: H,
  process_cov: ArrayView2<f64>,
  measurement_cov: ArrayView2<f64>,
  observation: ArrayView1<f64>,
  tuning: UkfTuning,
) -> Result<UkfState, StatsError>
where
  F: Fn(ArrayView1<f64>) -> Array1<f64>,
  H: Fn(ArrayView1<f64>) -> Array1<f64>,
{
  let UkfTuning { alpha, beta, kappa } = tuning;
  let n = prior.mean.len();
  assert!(
    prior.covariance.dim() == (n, n),
    "prior.covariance must satisfy `prior.covariance.dim() == (n, n)`, got prior.covariance.dim() = {:?}, n = {n}",
    prior.covariance.dim()
  );
  assert!(
    process_cov.dim() == (n, n),
    "process_cov must satisfy `process_cov.dim() == (n, n)`, got process_cov.dim() = {:?}, n = {n}",
    process_cov.dim()
  );
  let lambda = alpha * alpha * (n as f64 + kappa) - n as f64;
  assert_tuning(tuning, n, lambda);
  let scale = (n as f64 + lambda).sqrt();

  let chol =
    spd_cholesky_lower(prior.covariance.view()).ok_or(StatsError::Singular("prior covariance"))?;

  let n_sigma = 2 * n + 1;
  let mut sigma = Array2::<f64>::zeros((n_sigma, n));
  for j in 0..n {
    sigma[[0, j]] = prior.mean[j];
  }
  for i in 0..n {
    for j in 0..n {
      sigma[[1 + i, j]] = prior.mean[j] + scale * chol[[j, i]];
      sigma[[1 + n + i, j]] = prior.mean[j] - scale * chol[[j, i]];
    }
  }

  let mut wm = Array1::<f64>::zeros(n_sigma);
  let mut wc = Array1::<f64>::zeros(n_sigma);
  wm[0] = lambda / (n as f64 + lambda);
  wc[0] = wm[0] + 1.0 - alpha * alpha + beta;
  let w_other = 1.0 / (2.0 * (n as f64 + lambda));
  for i in 1..n_sigma {
    wm[i] = w_other;
    wc[i] = w_other;
  }

  let mut x_pred = Array2::<f64>::zeros((n_sigma, n));
  for i in 0..n_sigma {
    let row = sigma.row(i);
    let propagated = transition(row);
    assert!(
      propagated.len() == n,
      "transition's output must satisfy `output.len() == n`, got output.len() = {}, n = {n}",
      propagated.len()
    );
    for j in 0..n {
      x_pred[[i, j]] = propagated[j];
    }
  }

  let mut mean_pred = Array1::<f64>::zeros(n);
  for i in 0..n_sigma {
    for j in 0..n {
      mean_pred[j] += wm[i] * x_pred[[i, j]];
    }
  }

  let mut cov_pred = process_cov.to_owned();
  for i in 0..n_sigma {
    let mut diff = Array1::<f64>::zeros(n);
    for j in 0..n {
      diff[j] = x_pred[[i, j]] - mean_pred[j];
    }
    for r in 0..n {
      for c in 0..n {
        cov_pred[[r, c]] += wc[i] * diff[r] * diff[c];
      }
    }
  }

  let m = observation.len();
  assert!(
    measurement_cov.dim() == (m, m),
    "measurement_cov must satisfy `measurement_cov.dim() == (m, m)`, got measurement_cov.dim() = {:?}, m = {m}",
    measurement_cov.dim()
  );
  let mut z_pred_set = Array2::<f64>::zeros((n_sigma, m));
  for i in 0..n_sigma {
    let z = measurement(x_pred.row(i));
    assert!(
      z.len() == m,
      "measurement's output must satisfy `output.len() == m`, got output.len() = {}, m = {m}",
      z.len()
    );
    for j in 0..m {
      z_pred_set[[i, j]] = z[j];
    }
  }
  let mut z_mean = Array1::<f64>::zeros(m);
  for i in 0..n_sigma {
    for j in 0..m {
      z_mean[j] += wm[i] * z_pred_set[[i, j]];
    }
  }
  let mut s = measurement_cov.to_owned();
  for i in 0..n_sigma {
    let mut dz = Array1::<f64>::zeros(m);
    for j in 0..m {
      dz[j] = z_pred_set[[i, j]] - z_mean[j];
    }
    for r in 0..m {
      for c in 0..m {
        s[[r, c]] += wc[i] * dz[r] * dz[c];
      }
    }
  }
  let mut cross = Array2::<f64>::zeros((n, m));
  for i in 0..n_sigma {
    let mut dx = Array1::<f64>::zeros(n);
    for j in 0..n {
      dx[j] = x_pred[[i, j]] - mean_pred[j];
    }
    let mut dz = Array1::<f64>::zeros(m);
    for j in 0..m {
      dz[j] = z_pred_set[[i, j]] - z_mean[j];
    }
    for r in 0..n {
      for c in 0..m {
        cross[[r, c]] += wc[i] * dx[r] * dz[c];
      }
    }
  }
  let s_inv = inverse(s.view()).ok_or(StatsError::Singular("innovation covariance"))?;
  let kalman_gain = cross.dot(&s_inv);
  let innovation: Array1<f64> = (0..m)
    .map(|j| observation[j] - z_mean[j])
    .collect::<Vec<_>>()
    .into();
  let correction = kalman_gain.dot(&innovation);
  let mut mean_post = mean_pred.clone();
  for j in 0..n {
    mean_post[j] += correction[j];
  }
  let mut cov_post = cov_pred.clone();
  let kg_s = kalman_gain.dot(&s);
  let update = kg_s.dot(&kalman_gain.t());
  for r in 0..n {
    for c in 0..n {
      cov_post[[r, c]] -= update[[r, c]];
    }
  }
  Ok(UkfState {
    mean: mean_post,
    covariance: cov_post,
  })
}

#[track_caller]
#[inline(never)]
fn assert_tuning(tuning: UkfTuning, n: usize, lambda: f64) {
  let UkfTuning { alpha, beta, kappa } = tuning;
  assert!(
    alpha.is_finite(),
    "tuning.alpha must satisfy `tuning.alpha.is_finite()`, got tuning.alpha = {alpha:?}"
  );
  assert!(
    alpha > 0.0,
    "tuning.alpha must satisfy `tuning.alpha > 0.0`, got tuning.alpha = {alpha:?}"
  );
  assert!(
    beta.is_finite(),
    "tuning.beta must satisfy `tuning.beta.is_finite()`, got tuning.beta = {beta:?}"
  );
  assert!(
    kappa.is_finite(),
    "tuning.kappa must satisfy `tuning.kappa.is_finite()`, got tuning.kappa = {kappa:?}"
  );
  let spread = n as f64 + kappa;
  assert!(
    spread > 0.0,
    "n + tuning.kappa must satisfy `n + tuning.kappa > 0.0`, got n + tuning.kappa = {spread:?}, n = {n}"
  );
  let total = n as f64 + lambda;
  assert!(
    total.is_finite() && total > 0.0,
    "n + lambda must satisfy `(n + lambda).is_finite() && n + lambda > 0.0`, got n + lambda = {total:?}, lambda = {lambda:?}, n = {n}"
  );
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use ndarray::Array2;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::DistributionSampler;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  #[test]
  fn ukf_tracks_linear_random_walk() {
    let n = 200usize;
    let mut dist = SimdNormal::<f64>::new(0.0, 0.5).seeded(&Deterministic::new(1));
    let mut steps = vec![0.0_f64; n];
    dist.fill_slice(&mut steps);
    let mut obs_noise = SimdNormal::<f64>::new(0.0, 0.3).seeded(&Deterministic::new(2));
    let mut obs_eps = vec![0.0_f64; n];
    obs_noise.fill_slice(&mut obs_eps);
    let mut x_true = vec![0.0_f64; n];
    for i in 1..n {
      x_true[i] = x_true[i - 1] + steps[i];
    }
    let observations: Vec<f64> = (0..n).map(|i| x_true[i] + obs_eps[i]).collect();

    let mut state = UkfState {
      mean: Array1::from(vec![0.0_f64]),
      covariance: Array2::from_shape_vec((1, 1), vec![1.0_f64]).unwrap(),
    };
    let q = Array2::from_shape_vec((1, 1), vec![0.25_f64]).unwrap();
    let r = Array2::from_shape_vec((1, 1), vec![0.09_f64]).unwrap();
    let transition = |x: ArrayView1<f64>| Array1::from(vec![x[0]]);
    let measurement = |x: ArrayView1<f64>| Array1::from(vec![x[0]]);
    let mut errs = 0.0_f64;
    for t in 0..n {
      let y = Array1::from(vec![observations[t]]);
      state = unscented_kalman_step(
        &state,
        transition,
        measurement,
        q.view(),
        r.view(),
        y.view(),
        UkfTuning::default(),
      )
      .unwrap();
      errs += (state.mean[0] - x_true[t]).abs();
    }
    let mean_err = errs / n as f64;
    assert!(mean_err < 0.5, "mean error {mean_err}");
  }

  #[test]
  fn ukf_handles_two_dimensional_state() {
    let mut state = UkfState {
      mean: Array1::from(vec![0.0_f64, 0.0]),
      covariance: Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap(),
    };
    let q = Array2::from_shape_vec((2, 2), vec![0.1_f64, 0.0, 0.0, 0.1]).unwrap();
    let r = Array2::from_shape_vec((1, 1), vec![0.5_f64]).unwrap();
    let transition = |x: ArrayView1<f64>| Array1::from(vec![x[0] + x[1], x[1]]);
    let measurement = |x: ArrayView1<f64>| Array1::from(vec![x[0]]);
    for t in 0..50 {
      let y = Array1::from(vec![t as f64 * 0.5]);
      state = unscented_kalman_step(
        &state,
        transition,
        measurement,
        q.view(),
        r.view(),
        y.view(),
        UkfTuning::default(),
      )
      .unwrap();
      assert!(state.mean.iter().all(|v| v.is_finite()));
    }
  }

  #[test]
  fn a_prior_covariance_without_a_cholesky_factor_is_singular() {
    let prior = UkfState {
      mean: Array1::from(vec![0.0_f64, 0.0]),
      covariance: Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 1.0]).unwrap(),
    };
    let q = Array2::<f64>::eye(2);
    let r = Array2::<f64>::eye(1);
    let y = Array1::from(vec![0.0_f64]);
    let identity = |x: ArrayView1<f64>| x.to_owned();
    let first = |x: ArrayView1<f64>| Array1::from(vec![x[0]]);
    let err = unscented_kalman_step(
      &prior,
      identity,
      first,
      q.view(),
      r.view(),
      y.view(),
      UkfTuning::default(),
    )
    .unwrap_err();
    assert_eq!(err, StatsError::Singular("prior covariance"));
  }

  #[test]
  fn a_tuning_outside_its_range_panics_in_the_predicate_form() {
    let default = UkfTuning::default();
    let cases = [
      (
        UkfTuning {
          alpha: f64::NAN,
          ..default
        },
        "tuning.alpha must satisfy `tuning.alpha.is_finite()`, got tuning.alpha = NaN",
      ),
      (
        UkfTuning {
          alpha: 0.0,
          ..default
        },
        "tuning.alpha must satisfy `tuning.alpha > 0.0`, got tuning.alpha = 0.0",
      ),
      (
        UkfTuning {
          beta: f64::INFINITY,
          ..default
        },
        "tuning.beta must satisfy `tuning.beta.is_finite()`, got tuning.beta = inf",
      ),
      (
        UkfTuning {
          kappa: f64::NAN,
          ..default
        },
        "tuning.kappa must satisfy `tuning.kappa.is_finite()`, got tuning.kappa = NaN",
      ),
      (
        UkfTuning {
          kappa: -2.0,
          ..default
        },
        "n + tuning.kappa must satisfy `n + tuning.kappa > 0.0`, got n + tuning.kappa = 0.0, n = 2",
      ),
      (
        UkfTuning {
          alpha: 1e-9,
          ..default
        },
        "n + lambda must satisfy `(n + lambda).is_finite() && n + lambda > 0.0`, got n + lambda = 0.0, lambda = -2.0, n = 2",
      ),
      (
        UkfTuning {
          alpha: 1e155,
          ..default
        },
        "n + lambda must satisfy `(n + lambda).is_finite() && n + lambda > 0.0`, got n + lambda = inf, lambda = inf, n = 2",
      ),
    ];
    for (tuning, message) in cases {
      let payload = std::panic::catch_unwind(|| {
        let prior = UkfState {
          mean: Array1::from(vec![0.0_f64, 0.0]),
          covariance: Array2::<f64>::eye(2),
        };
        let identity = |x: ArrayView1<f64>| x.to_owned();
        let first = |x: ArrayView1<f64>| Array1::from(vec![x[0]]);
        let (q, r) = (Array2::<f64>::eye(2), Array2::<f64>::eye(1));
        let y = Array1::from(vec![0.5_f64]);
        let _ = unscented_kalman_step(
          &prior,
          identity,
          first,
          q.view(),
          r.view(),
          y.view(),
          tuning,
        );
      })
      .expect_err("the step accepted an invalid tuning");
      let text = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a text panic payload");
      assert_eq!(text, message);
    }
  }

  #[test]
  fn a_singular_innovation_covariance_is_singular() {
    let prior = UkfState {
      mean: Array1::from(vec![0.0_f64]),
      covariance: Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
    };
    let q = Array2::<f64>::zeros((1, 1));
    let r = Array2::<f64>::zeros((1, 1));
    let y = Array1::from(vec![0.0_f64]);
    let identity = |x: ArrayView1<f64>| x.to_owned();
    let constant = |_: ArrayView1<f64>| Array1::from(vec![1.0_f64]);
    let err = unscented_kalman_step(
      &prior,
      identity,
      constant,
      q.view(),
      r.view(),
      y.view(),
      UkfTuning::default(),
    )
    .unwrap_err();
    assert_eq!(err, StatsError::Singular("innovation covariance"));
  }
}
