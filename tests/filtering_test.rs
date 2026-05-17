//! Integration tests for the `stats::filtering` module.

use ndarray::Array1;
#[cfg(feature = "openblas")]
use ndarray::Array2;
use ndarray::ArrayView1;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::simd_rng::SimdRng;
use stochastic_rs::stats::filtering::ParticleFilter;
#[cfg(feature = "openblas")]
use stochastic_rs::stats::filtering::UkfState;
use stochastic_rs::stats::filtering::random_walk_metropolis;
#[cfg(feature = "openblas")]
use stochastic_rs::stats::filtering::unscented_kalman_step;

#[test]
fn particle_filter_with_systematic_resampling_runs_to_completion() {
  let init_dist = SimdNormal::<f64>::new(0.0, 1.0, &stochastic_rs_core::simd_rng::Deterministic::new(1));
  let init_fn = move |_rng: &mut SimdRng| {
    let mut a = [0.0_f64];
    init_dist.fill_slice_fast(&mut a);
    Array1::from(vec![a[0]])
  };
  let trans_dist = SimdNormal::<f64>::new(0.0, 1.0, &stochastic_rs_core::simd_rng::Deterministic::new(2));
  let transition = move |prev: ArrayView1<f64>, _rng: &mut SimdRng| {
    let mut a = [0.0_f64];
    trans_dist.fill_slice_fast(&mut a);
    Array1::from(vec![prev[0] + a[0]])
  };
  let log_obs = |x: ArrayView1<f64>, y: ArrayView1<f64>| {
    let z = (y[0] - x[0]) / 0.5;
    -0.5 * z * z
  };
  let mut pf = ParticleFilter::new(200, init_fn, transition, log_obs, 5);
  for t in 0..50 {
    let y = Array1::from(vec![t as f64 * 0.1]);
    pf.step(y.view());
  }
  let m = pf.mean_state();
  assert!(m[0].is_finite());
  assert!(pf.effective_sample_size() > 0.0);
}

#[test]
fn metropolis_chain_concentrates_around_mode_of_skewed_target() {
  let init = Array1::from(vec![1.0_f64]);
  let log_target = |x: ArrayView1<f64>| {
    let m = 2.5;
    -0.5 * ((x[0] - m) / 0.5).powi(2)
  };
  let scale = Array1::from(vec![0.7_f64]);
  let res = random_walk_metropolis(init.view(), log_target, scale.view(), 10_000, 1_000, 23);
  let mean = res.samples.column(0).iter().sum::<f64>() / 10_000.0;
  assert!((mean - 2.5).abs() < 0.1);
}

#[cfg(feature = "openblas")]
#[test]
fn ukf_remains_finite_under_long_run() {
  let mut state = UkfState {
    mean: Array1::from(vec![0.0_f64, 0.0]),
    covariance: Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap(),
  };
  let q = Array2::from_shape_vec((2, 2), vec![0.05_f64, 0.0, 0.0, 0.05]).unwrap();
  let r = Array2::from_shape_vec((1, 1), vec![0.5_f64]).unwrap();
  let transition = |x: ArrayView1<f64>| Array1::from(vec![x[0] + x[1] * 0.1, x[1]]);
  let measurement = |x: ArrayView1<f64>| Array1::from(vec![x[0]]);
  for t in 0..500 {
    let y = Array1::from(vec![(t as f64) * 0.05]);
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
    assert!(state.covariance.iter().all(|v| v.is_finite()));
  }
}
