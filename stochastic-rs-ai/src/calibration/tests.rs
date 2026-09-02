use candle_core::Device;
use ndarray::Array2;

use super::*;
use crate::volatility::common::TrainConfig;
use crate::volatility::common::synthetic_surface_dataset;
use crate::volatility::heston;
use crate::volatility::rbergomi;

fn trained_heston(seed: u64) -> HestonNn {
  let (params, surfaces) = synthetic_surface_dataset(
    &heston::PARAM_LB,
    &heston::PARAM_UB,
    256,
    heston::OUTPUT_DIM,
    seed,
  );
  let mut model = HestonNn::new(&Device::Cpu).unwrap();
  let cfg = TrainConfig {
    epochs: 30,
    random_seed: seed,
    ..TrainConfig::default()
  };
  model.train(&params, &surfaces, &cfg).unwrap();
  model
}

fn box_point(lb: &[f32], ub: &[f32], fractions: &[f64]) -> Vec<f64> {
  lb.iter()
    .zip(ub)
    .zip(fractions)
    .map(|((&l, &u), &f)| l as f64 + f * (u as f64 - l as f64))
    .collect()
}

/// The reverse-mode Jacobian agrees with central finite differences of the
/// plain prediction (both in parameter / implied-vol units).
#[test]
fn network_jacobian_matches_finite_differences() {
  let model = trained_heston(3);
  let theta = box_point(
    &heston::PARAM_LB,
    &heston::PARAM_UB,
    &[0.4, 0.6, 0.3, 0.7, 0.5],
  );
  let theta32: Vec<f32> = theta.iter().map(|&v| v as f32).collect();
  let (surface, jacobian) = model.nn().predict_surface_with_jacobian(&theta32).unwrap();
  assert_eq!(surface.len(), heston::OUTPUT_DIM);
  assert_eq!(jacobian.dim(), (heston::OUTPUT_DIM, heston::INPUT_DIM));
  let plain = model
    .predict_surface(&[theta32[0], theta32[1], theta32[2], theta32[3], theta32[4]])
    .unwrap();
  assert!(
    surface
      .iter()
      .zip(&plain)
      .all(|(a, b)| (a - b).abs() < 1e-6)
  );
  let scale = jacobian.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
  for j in 0..heston::INPUT_DIM {
    let h = 1e-3 * (heston::PARAM_UB[j] - heston::PARAM_LB[j]);
    let mut up = theta32.clone();
    let mut down = theta32.clone();
    up[j] += h;
    down[j] -= h;
    let f_up = model.nn().predict_surface(&up).unwrap();
    let f_down = model.nn().predict_surface(&down).unwrap();
    for k in 0..heston::OUTPUT_DIM {
      let fd = (f_up[k] - f_down[k]) / (2.0 * h);
      assert!(
        (jacobian[(k, j)] - fd).abs() < 2e-2 * scale + 1e-3,
        "k {k} j {j}: autodiff {} vs fd {fd}",
        jacobian[(k, j)]
      );
    }
  }
}

/// Calibrating the surrogate to its own surface at a known parameter point
/// recovers that point: the Jacobian, the coordinate transform and the
/// optimiser are consistent.
#[test]
fn self_consistent_calibration_recovers_the_parameters() {
  let model = trained_heston(5);
  let truth = box_point(
    &heston::PARAM_LB,
    &heston::PARAM_UB,
    &[0.35, 0.55, 0.6, 0.4, 0.65],
  );
  let truth32: Vec<f32> = truth.iter().map(|&v| v as f32).collect();
  let target: Vec<f64> = model
    .nn()
    .predict_surface(&truth32)
    .unwrap()
    .iter()
    .map(|&v| v as f64)
    .collect();
  let calibrator = SurrogateCalibrator::new(&model, target).unwrap();
  let result = calibrator.calibrate(None).unwrap();
  assert!(result.converged, "{}", result.message);
  assert!(result.in_bounds);
  assert!(result.rmse < 1e-6, "rmse {}", result.rmse);
  assert!(result.max_error < 1e-5);
  for (j, (fitted, expected)) in result.params.iter().zip(&truth).enumerate() {
    let range = (heston::PARAM_UB[j] - heston::PARAM_LB[j]) as f64;
    assert!(
      (fitted - expected).abs() < 2e-3 * range,
      "parameter {j}: {fitted} vs {expected}"
    );
  }
  assert_eq!(result.iterations(), Some(result.evaluations));
  assert!(result.evaluations > 0);
}

/// The typed Heston bridge maps the surrogate order `[v0, ρ, σ, θ, κ]` onto
/// `HestonParams` and hands the fitted parameters to the Fourier pricer.
#[test]
fn heston_bridge_maps_the_parameter_order_and_builds_the_pricer() {
  let model = trained_heston(7);
  let truth = HestonParams {
    v0: 0.02,
    kappa: 3.0,
    theta: 0.05,
    sigma: 0.4,
    rho: -0.6,
  };
  let as_surrogate = heston_params_to_surrogate(&truth);
  assert_eq!(as_surrogate, vec![0.02, -0.6, 0.4, 0.05, 3.0]);
  assert_eq!(heston_params_from_surrogate(&as_surrogate), truth);
  let target: Vec<f64> = model
    .nn()
    .predict_surface(&as_surrogate.iter().map(|&v| v as f32).collect::<Vec<_>>())
    .unwrap()
    .iter()
    .map(|&v| v as f64)
    .collect();
  let calibrator = HestonSurrogateCalibrator::new(&model, target).unwrap();
  let result = calibrator.calibrate(Some(truth.clone())).unwrap();
  assert!(result.converged() && result.rmse() < 1e-6);
  assert!((result.params().kappa - 3.0).abs() < 0.05);
  let pricer = result.to_model(0.01, 0.0);
  assert!((pricer.kappa - result.params.kappa).abs() < 1e-12);
  assert!((pricer.rho - result.params.rho).abs() < 1e-12);
}

/// The rough Bergomi bridge maps `[ξ₀, η, ρ, H]` onto a flat-ξ₀
/// `RBergomiParams`, both ways.
#[test]
fn rbergomi_bridge_round_trips_the_parameter_order() {
  let params = RBergomiParams {
    hurst: 0.1,
    rho: -0.7,
    eta: 1.9,
    xi0: RBergomiXi0::Constant(0.04),
  };
  let v = rbergomi_params_to_surrogate(&params);
  assert_eq!(v, vec![0.04, 1.9, -0.7, 0.1]);
  let back = rbergomi_params_from_surrogate(&v);
  assert_eq!(back.hurst, 0.1);
  assert_eq!(back.rho, -0.7);
  assert_eq!(back.eta, 1.9);
  assert!(matches!(back.xi0, RBergomiXi0::Constant(x) if x == 0.04));
  let (p, s) = synthetic_surface_dataset(
    &rbergomi::PARAM_LB,
    &rbergomi::PARAM_UB,
    96,
    rbergomi::OUTPUT_DIM,
    2,
  );
  let mut model = RBergomiNn::new(&Device::Cpu).unwrap();
  let cfg = TrainConfig {
    epochs: 5,
    ..TrainConfig::default()
  };
  model.train(&p, &s, &cfg).unwrap();
  let target: Vec<f64> = model
    .nn()
    .predict_surface(&v.iter().map(|&x| x as f32).collect::<Vec<_>>())
    .unwrap()
    .iter()
    .map(|&x| x as f64)
    .collect();
  let result = RBergomiSurrogateCalibrator::new(&model, target)
    .unwrap()
    .calibrate(Some(params))
    .unwrap();
  assert!(result.rmse() < 1e-5, "{}", result.rmse());
  let _pricer: RBergomiPricer = result.to_model(0.0, 0.0);
}

#[test]
fn input_validation_rejects_mismatched_shapes() {
  let model = trained_heston(9);
  assert!(SurrogateCalibrator::new(&model, vec![0.2; 10]).is_err());
  let calibrator = SurrogateCalibrator::new(&model, vec![0.2; heston::OUTPUT_DIM]).unwrap();
  assert!(calibrator.with_weights(vec![1.0; 3]).is_err());
  let calibrator = SurrogateCalibrator::new(&model, vec![0.2; heston::OUTPUT_DIM]).unwrap();
  assert!(calibrator.with_initial(vec![0.0; 2]).is_err());
  let nan = {
    let mut m = vec![0.2; heston::OUTPUT_DIM];
    m[3] = f64::NAN;
    m
  };
  assert!(SurrogateCalibrator::new(&model, nan).is_err());
  let _ = Array2::<f32>::zeros((1, 1));
}
