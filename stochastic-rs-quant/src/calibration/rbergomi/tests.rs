use super::*;
use crate::traits::Calibrator;

#[test]
fn test_empirical_wasserstein_1_matches_simple_case() {
  let a = vec![1.0, 2.0, 3.0];
  let b = vec![2.0, 3.0, 4.0];
  let w1 = empirical_wasserstein_1(&a, &b);
  assert!((w1 - 1.0).abs() < 1e-12);
}

#[test]
fn test_rbergomi_calibration_reduces_loss_on_synthetic_data() {
  let true_params = RBergomiParams {
    hurst: 0.12,
    rho: -0.72,
    eta: 1.6,
    xi0: RBergomiXi0::Constant(0.04),
  };

  let maturities = [0.25, 0.5, 1.0];
  let mut market_slices = Vec::with_capacity(maturities.len());
  for (i, &t) in maturities.iter().enumerate() {
    let market_samples = simulate_rbergomi_terminal_samples(
      &true_params,
      100.0,
      0.01,
      0.0,
      t,
      512,
      96,
      12,
      7_777 + i as u64,
    );
    market_slices.push(RBergomiMarketSlice {
      maturity: t,
      terminal_samples: market_samples,
    });
  }

  let init_params = RBergomiParams {
    hurst: 0.30,
    rho: -0.10,
    eta: 0.80,
    xi0: RBergomiXi0::Constant(0.02),
  };

  let config = RBergomiCalibrationConfig {
    paths: 512,
    steps_per_year: 96,
    msoe_terms: 12,
    max_iters: 12,
    learning_rate: 0.08,
    finite_diff_eps: 5e-3,
    adam_beta1: 0.9,
    adam_beta2: 0.99,
    adam_eps: 1e-8,
    random_seed: 123_456,
    stop_loss: None,
    improvement_tol: 1e-5,
  };

  let calibrator = RBergomiCalibrator::new(
    100.0,
    0.01,
    init_params.clone(),
    market_slices,
    config,
    true,
  )
  .expect("RBergomi calibrator construction must succeed in test");

  let result = calibrator.calibrate(None).unwrap();

  println!(
    "rBergomi calibration: initial_loss={:.6}, final_loss={:.6}, iterations={}",
    result.initial_loss, result.final_loss, result.iterations
  );
  println!("initial params: {:?}", result.initial_params);
  println!("estimated params: {:?}", result.calibrated_params);
  println!("per-maturity W1: {:?}", result.maturity_losses);

  assert!(result.final_loss <= result.initial_loss);
  assert!(result.calibrated_params.hurst > 0.0 && result.calibrated_params.hurst < 0.5);
  assert!(result.calibrated_params.rho.abs() < 1.0);
  assert!(result.calibrated_params.eta > 0.0);

  if let RBergomiXi0::Constant(xi0_hat) = result.calibrated_params.xi0 {
    assert!(xi0_hat > 0.0);
  } else {
    panic!("Expected constant xi0 in this test");
  }
}
