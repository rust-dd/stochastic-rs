//! End-to-end calibration demo: BSM and Heston.
//!
//! Walks through the full v2 calibration pipeline:
//!   `Calibrator::calibrate` → `CalibrationResult` → `ToModel::to_model` →
//!   `ModelPricer::price_call`. We synthesise a market from a known parameter
//!   set, calibrate, then re-price to verify recovery.
//!
//! Run with:
//!   cargo run --example calibration_demo

use nalgebra::DVector;
use stochastic_rs::prelude::*;
use stochastic_rs::quant::OptionType;
use stochastic_rs::quant::calibration::bsm::BSMCalibrator;
use stochastic_rs::quant::calibration::bsm::BSMParams;
use stochastic_rs::quant::calibration::heston::HestonCalibrator;
use stochastic_rs::quant::calibration::heston::HestonParams;
use stochastic_rs::quant::calibration::levy::MarketSlice;
use stochastic_rs::quant::pricing::fourier::HestonFourier;

fn main() {
  println!("=== stochastic-rs calibration demo ===\n");
  bsm_demo();
  println!();
  heston_demo();
}

/// BSM: synthesise prices from a known sigma, calibrate, recover.
fn bsm_demo() {
  println!("[BSM] flat-vol calibration (single maturity)");

  let s0 = 100.0_f64;
  let r = 0.03_f64;
  let q = 0.0_f64;
  let tau = 0.5_f64;
  let sigma_true = 0.22_f64;
  let strikes = vec![80.0_f64, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];

  // Synthesise call prices from the closed-form chf via Gil-Pelaez.
  let model_true = stochastic_rs::quant::pricing::fourier::BSMFourier {
    sigma: sigma_true,
    r,
    q,
  };
  let market: Vec<f64> = strikes
    .iter()
    .map(|&k| model_true.price_call(s0, k, r, q, tau))
    .collect();

  let calibrator = BSMCalibrator::new(
    BSMParams { v: 0.30 },
    DVector::from_vec(market.clone()),
    DVector::from_vec(vec![s0; strikes.len()]),
    DVector::from_vec(strikes.clone()),
    r,
    None,
    None,
    Some(q),
    tau,
    OptionType::Call,
  );

  let res = calibrator
    .calibrate(None)
    .expect("BSM calibration should succeed");

  println!(
    "  σ_true = {:.4}, σ_calibrated = {:.4}, |Δσ| = {:.2e}",
    sigma_true,
    res.params().v,
    (res.params().v - sigma_true).abs(),
  );
  println!(
    "  RMSE = {:.4e}, converged = {}",
    res.rmse(),
    res.converged()
  );
  assert!(
    (res.params().v - sigma_true).abs() < 1e-4,
    "BSM should recover σ to ~1e-4"
  );

  let model = res.to_model(r, q);
  let max_repricing_err = strikes
    .iter()
    .zip(market.iter())
    .map(|(&k, &px)| (model.price_call(s0, k, r, q, tau) - px).abs())
    .fold(0.0_f64, f64::max);
  println!("  max re-pricing error = {max_repricing_err:.2e}");
  assert!(max_repricing_err < 1e-3);
}

/// Heston: synthesise a multi-maturity IV surface from a known parameter set,
/// calibrate jointly across maturities, recover.
///
/// Multi-maturity is important: with a single maturity Heston is weakly
/// identified (different (v0, κ, θ, σ, ρ) tuples can fit the same smile).
/// Adding maturities lets the term-structure pin down κ and θ.
fn heston_demo() {
  println!("[Heston] joint multi-maturity surface calibration");

  let s0 = 100.0_f64;
  let r = 0.025_f64;
  let q = 0.0_f64;

  let true_params = HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.30,
    rho: -0.6,
  };
  let model_true = HestonFourier {
    v0: true_params.v0,
    kappa: true_params.kappa,
    theta: true_params.theta,
    sigma: true_params.sigma,
    rho: true_params.rho,
    r,
    q,
  };

  let strikes = vec![80.0_f64, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];
  let maturities = [0.25_f64, 0.5, 1.0, 2.0];

  let slices: Vec<MarketSlice> = maturities
    .iter()
    .map(|&t| MarketSlice {
      strikes: strikes.clone(),
      prices: strikes
        .iter()
        .map(|&k| model_true.price_call(s0, k, r, q, t))
        .collect(),
      is_call: vec![true; strikes.len()],
      t,
    })
    .collect();

  // Initial guess deliberately off the true values to test the optimiser.
  let initial = HestonParams {
    v0: 0.05,
    kappa: 2.5,
    theta: 0.06,
    sigma: 0.5,
    rho: -0.3,
  };
  let calibrator = HestonCalibrator::from_slices(
    Some(initial),
    &slices,
    s0,
    r,
    Some(q),
    OptionType::Call,
    false,
  );

  let res = calibrator
    .calibrate(None)
    .expect("Heston calibration should succeed");

  let p = res.params();
  println!(
    "  true:        v0={:.4} κ={:.3} θ={:.4} σ={:.3} ρ={:+.3}",
    true_params.v0, true_params.kappa, true_params.theta, true_params.sigma, true_params.rho
  );
  println!(
    "  calibrated:  v0={:.4} κ={:.3} θ={:.4} σ={:.3} ρ={:+.3}",
    p.v0, p.kappa, p.theta, p.sigma, p.rho
  );
  println!(
    "  RMSE = {:.4e}, converged = {}",
    res.rmse(),
    res.converged()
  );

  // ToModel → ModelPricer round-trip on the full surface.
  let model = res.to_model(r, q);
  let mut max_err = 0.0_f64;
  for slice in &slices {
    for (&k, &px) in slice.strikes.iter().zip(slice.prices.iter()) {
      let priced = model.price_call(s0, k, r, q, slice.t);
      max_err = max_err.max((priced - px).abs());
    }
  }
  println!("  max re-pricing error across surface = {max_err:.2e}");
}
