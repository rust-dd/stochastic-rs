use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::diffusion::fou::Fou;

use super::*;
use crate::traits::ProcessExt;

fn bounded_problem() -> WhittleProblem {
  WhittleProblem {
    pgram: vec![0.1; 16],
    gamma: vec![0.1; 32],
    m: 72,
    n: 32,
    psi: 1e-5,
    k_trunc: 20,
    j_max: 5,
    lower: vec![0.005, 0.01],
    upper: vec![0.495, 12.0],
  }
}

#[test]
fn bounded_whittle_cost_and_gradient_use_feasible_parameters() {
  let problem = bounded_problem();
  for (outside, boundary) in [
    (vec![0.6, 1.0], vec![0.495, 1.0]),
    (vec![-0.1, 1.0], vec![0.005, 1.0]),
    (vec![0.3, 15.0], vec![0.3, 12.0]),
  ] {
    assert_eq!(problem.eval(&outside), problem.eval(&boundary));
    let expected = problem.gradient(&boundary).unwrap();
    assert!(expected.iter().any(|value| value.abs() > 1e-6));
    assert_eq!(problem.gradient(&outside).unwrap(), expected);
  }
}

#[test]
fn bounded_whittle_refinement_returns_feasible_parameters_and_cost() {
  let problem = bounded_problem();
  for initial in [[0.2, 1.0], [0.6, 15.0], [-0.1, -1.0]] {
    let (h, v, cost) = run_lbfgs(&problem, initial[0], initial[1]);
    assert!((problem.lower[0]..=problem.upper[0]).contains(&h));
    assert!((problem.lower[1]..=problem.upper[1]).contains(&v));
    assert!(cost.is_finite());
    assert_eq!(cost, problem.eval(&[h, v]));
  }
}

fn simulate_log_rv(true_h: f64, m: usize, n_days: usize, delta: f64, seed: u64) -> Vec<f64> {
  let fou = Fou::new(
    true_h,
    0.001,
    -3.2,
    1.0,
    n_days + 1,
    Some(-3.2),
    Some(n_days as f64 * delta),
    Deterministic::new(seed),
  );
  let log_vol_sq: Array1<f64> = fou.sample();
  let mut log_rv = vec![0.0_f64; n_days];
  let mut normals = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(0x7717));
  for day in 0..n_days {
    let sigma = log_vol_sq[day].exp().sqrt();
    let dt = delta / m as f64;
    let mut rv = 0.0;
    for _ in 0..m {
      let z = normals.sample();
      rv += (sigma * dt.sqrt() * z).powi(2);
    }
    log_rv[day] = rv.max(1e-20).ln();
  }
  log_rv
}

#[test]
fn estimate_h_from_simulated_rv() {
  let true_h = 0.3_f64;
  let log_rv = simulate_log_rv(true_h, 72, 500, 1.0 / 250.0, 42);
  let result = Whittle::new(72, 1.0 / 250.0)
    .fit(ArrayView1::from(&log_rv))
    .unwrap();
  assert!(
    (result.hurst - true_h).abs() < 0.1,
    "H={:.3}, true={true_h}",
    result.hurst
  );
}

#[test]
fn rough_vs_smooth_distinguished() {
  let whittle = Whittle::new(72, 1.0 / 250.0);
  let log_rv_rough = simulate_log_rv(0.1, 72, 500, 1.0 / 250.0, 77);
  let log_rv_smooth = simulate_log_rv(0.45, 72, 500, 1.0 / 250.0, 88);
  let h_rough = whittle.fit(ArrayView1::from(&log_rv_rough)).unwrap().hurst;
  let h_smooth = whittle.fit(ArrayView1::from(&log_rv_smooth)).unwrap().hurst;
  assert!(
    h_rough < h_smooth,
    "rough H={h_rough:.3} should be < smooth H={h_smooth:.3}"
  );
}

/// `fit` is the estimator's one entry; the `HurstEstimator` trait is for path self-similarity, not latent volatility.
#[test]
fn fit_reports_the_fukasawa_estimate_and_rejects_short_or_non_finite_data() {
  let whittle = Whittle::new(72, 1.0 / 250.0);
  let short = Array1::<f64>::linspace(-1.0, 1.0, 10);
  assert_eq!(
    whittle.fit(short.view()).unwrap_err(),
    StatsError::InsufficientData {
      needed: 30,
      got: 10
    }
  );
  let mut with_nan = Array1::<f64>::linspace(-1.0, 1.0, 40);
  with_nan[5] = f64::NAN;
  assert!(matches!(
    whittle.fit(with_nan.view()),
    Err(StatsError::InvalidObservation { index: 5, .. })
  ));
  let log_rv = simulate_log_rv(0.1, 72, 400, 1.0 / 250.0, 3);
  let fit = whittle.fit(ArrayView1::from(&log_rv)).unwrap();
  assert!(
    (fit.hurst - 0.1).abs() < 0.08 && fit.eta > 0.0 && fit.nobs == 400,
    "{fit:?}"
  );
}
