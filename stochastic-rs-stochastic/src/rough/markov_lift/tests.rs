use ndarray::Array2;

use super::MarkovLift;
use crate::rough::kernel::RlKernel;

#[test]
fn trivial_drift_zero_diffusion_stays_at_x0() {
  let kernel = RlKernel::<f64>::new(0.15, 30);
  let dt = 0.01_f64;
  let step = MarkovLift::new(kernel, dt);
  let dw = vec![0.0_f64; 50];
  let path = step.simulate(0.42, |_| 0.0, |_| 0.0, &dw);
  for v in path.iter() {
    assert!((*v - 0.42).abs() < 1e-12);
  }
}

#[test]
fn constant_drift_matches_mittag_leffler_linear_case() {
  let hurst = 0.3_f64;
  let c = 1.5_f64;
  let n = 201;
  let total_t = 1.0_f64;
  let dt = total_t / (n as f64 - 1.0);
  let kernel = RlKernel::<f64>::new(hurst, 40);
  let step = MarkovLift::new(kernel, dt);
  let dw = vec![0.0_f64; n - 1];

  let path = step.simulate(0.0, |_| c, |_| 0.0, &dw);

  let exponent = hurst + 0.5;
  let gamma_h_three_half = stochastic_rs_distributions::special::gamma(hurst + 1.5);
  for i in 1..n {
    let t = dt * i as f64;
    let truth = c * t.powf(exponent) / gamma_h_three_half;
    let rel = (path[i] - truth).abs() / truth.abs().max(1e-9);
    assert!(
      rel < 2e-2,
      "i={i} t={t} got={} truth={truth} rel={rel}",
      path[i]
    );
  }
}

#[test]
fn f32_path_is_finite() {
  let kernel = RlKernel::<f32>::new(0.25_f32, 32);
  let dt = 0.005_f32;
  let step = MarkovLift::new(kernel, dt);
  let dw: Vec<f32> = (0..100).map(|i| ((i as f32 * 0.1).sin()) * 0.05).collect();
  let path = step.simulate(0.3_f32, |x| 0.5 * (1.0 - x), |_| 0.2, &dw);
  assert!(path.iter().all(|v| v.is_finite()));
}

/// The batch simulator must produce the same path as repeated single-path
/// runs with the matching per-row increments.
#[test]
fn batch_matches_single_path_row_by_row() {
  let hurst = 0.22_f64;
  let n = 65;
  let m = 7;
  let dt = 1.0_f64 / (n as f64 - 1.0);
  let kernel = RlKernel::<f64>::new(hurst, 30);
  let step = MarkovLift::new(kernel, dt);

  let mut dw = Array2::<f64>::zeros((m, n - 1));
  for p in 0..m {
    for i in 0..n - 1 {
      dw[[p, i]] = ((p as f64 + 1.0) * 0.13 + (i as f64) * 0.027).sin() * 0.02;
    }
  }

  let batch = step.simulate_batch(0.4, |x| 0.6 * (1.0 - x), |_| 0.15, dw.view());
  assert_eq!(batch.dim(), (m, n));

  for p in 0..m {
    let row = dw.row(p).to_vec();
    let single = step.simulate(0.4, |x| 0.6 * (1.0 - x), |_| 0.15, row.as_slice());
    for i in 0..n {
      let diff = (batch[[p, i]] - single[i]).abs();
      assert!(
        diff < 1e-12,
        "p={p} i={i} batch={} single={} diff={diff}",
        batch[[p, i]],
        single[i]
      );
    }
  }
}
