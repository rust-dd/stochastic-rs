//! Unscented Kalman Filter (UKF) — single forward step.
//!
//! Approximates the moments of a non-linear stochastic state-space model by
//! propagating $2n+1$ deterministically-chosen sigma points through the
//! transition $f$ and measurement $h$ functions, then computing weighted
//! sample mean and covariance.
//!
//! Reference: Julier, Uhlmann, "Unscented Filtering and Nonlinear Estimation",
//! Proceedings of the IEEE, 92(3), 401-422 (2004).
//! DOI: 10.1109/JPROC.2003.823141
//!
//! Reference: Wan, van der Merwe, "The Unscented Kalman Filter for Nonlinear
//! Estimation", IEEE Adaptive Systems for Signal Processing, Communications,
//! and Control Symposium 2000, 153-158. DOI: 10.1109/ASSPCC.2000.882463

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray_linalg::Cholesky;
use ndarray_linalg::Inverse;
use ndarray_linalg::UPLO;

/// State estimate (mean + covariance) on entry to / exit from a UKF step.
#[derive(Debug, Clone)]
pub struct UkfState {
  /// State posterior mean.
  pub mean: Array1<f64>,
  /// State posterior covariance.
  pub covariance: Array2<f64>,
}

/// One predict-update cycle of the Unscented Kalman Filter.
///
/// `prior`: state at time $t-1$.
/// `transition`: $f: \mathbb R^{n_x} \to \mathbb R^{n_x}$ (deterministic; the
/// process noise is added via `process_cov`).
/// `measurement`: $h: \mathbb R^{n_x} \to \mathbb R^{n_y}$.
/// `process_cov`, `measurement_cov`: $Q$ and $R$.
/// `observation`: $y_t$.
/// `alpha`, `beta`, `kappa`: standard UKF tuning parameters; defaults
/// $(0.001, 2, 0)$ are a sensible starting point.
pub fn unscented_kalman_step<F, H>(
  prior: &UkfState,
  transition: F,
  measurement: H,
  process_cov: ArrayView2<f64>,
  measurement_cov: ArrayView2<f64>,
  observation: ArrayView1<f64>,
  alpha: f64,
  beta: f64,
  kappa: f64,
) -> UkfState
where
  F: Fn(ArrayView1<f64>) -> Array1<f64>,
  H: Fn(ArrayView1<f64>) -> Array1<f64>,
{
  let n = prior.mean.len();
  assert_eq!(prior.covariance.dim(), (n, n));
  assert_eq!(process_cov.dim(), (n, n));
  let lambda = alpha * alpha * (n as f64 + kappa) - n as f64;
  let scale = (n as f64 + lambda).sqrt();

  let chol = prior
    .covariance
    .cholesky(UPLO::Lower)
    .expect("UKF prior covariance not PD");

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
    assert_eq!(propagated.len(), n);
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
  assert_eq!(measurement_cov.dim(), (m, m));
  let mut z_pred_set = Array2::<f64>::zeros((n_sigma, m));
  for i in 0..n_sigma {
    let z = measurement(x_pred.row(i));
    assert_eq!(z.len(), m);
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
  let s_inv = s.inv().expect("UKF innovation covariance singular");
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
  UkfState {
    mean: mean_post,
    covariance: cov_post,
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use ndarray::Array2;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  #[test]
  fn ukf_tracks_linear_random_walk() {
    let n = 200usize;
    let dist = SimdNormal::<f64>::with_seed(0.0, 0.5, 1);
    let mut steps = vec![0.0_f64; n];
    dist.fill_slice_fast(&mut steps);
    let obs_noise = SimdNormal::<f64>::with_seed(0.0, 0.3, 2);
    let mut obs_eps = vec![0.0_f64; n];
    obs_noise.fill_slice_fast(&mut obs_eps);
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
        0.001,
        2.0,
        0.0,
      );
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
        0.001,
        2.0,
        0.0,
      );
      assert!(state.mean.iter().all(|v| v.is_finite()));
    }
  }
}
