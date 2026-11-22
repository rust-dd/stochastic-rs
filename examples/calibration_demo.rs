//! BSM and Heston calibration end to end: `Calibrator::calibrate`, `ToModel::to_model` and
//! `ModelPricer::price_call` recover known parameters (`cargo run --example calibration_demo`).

use stochastic_rs::prelude::*;
use stochastic_rs::quant::OptionQuote;
use stochastic_rs::quant::OptionType;
use stochastic_rs::quant::calibration::CalibrationOptions;
use stochastic_rs::quant::calibration::bsm::BsmCalibrator;
use stochastic_rs::quant::calibration::bsm::BsmParams;
use stochastic_rs::quant::calibration::heston::HestonCalibrator;
use stochastic_rs::quant::pricing::fourier::HestonFourier;
use stochastic_rs::quant::pricing::heston::HestonParams;

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
  let strikes = [80.0_f64, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];

  // Synthesise call prices from the closed-form chf via Gil-Pelaez.
  let model_true = stochastic_rs::quant::pricing::fourier::BsmFourier { sigma: sigma_true };
  let market: Vec<f64> = strikes
    .iter()
    .map(|&k| model_true.price_call(s0, k, r, q, tau))
    .collect();

  let quotes = strikes
    .iter()
    .zip(&market)
    .map(|(&k, &price)| OptionQuote::new(k, tau, price, OptionType::Call))
    .collect();
  let calibrator = BsmCalibrator::new(s0, r, q, quotes).with_start(BsmParams { sigma: 0.30 });

  let res = calibrator
    .calibrate(None)
    .expect("BSM calibration should succeed");

  println!(
    "  σ_true = {:.4}, σ_calibrated = {:.4}, |Δσ| = {:.2e}",
    sigma_true,
    res.params().sigma,
    (res.params().sigma - sigma_true).abs(),
  );
  println!(
    "  RMSE = {:.4e}, converged = {}",
    res.rmse(),
    res.converged()
  );
  assert!(
    (res.params().sigma - sigma_true).abs() < 1e-4,
    "BSM should recover σ to ~1e-4"
  );

  let model = res.to_model();
  let max_repricing_err = f64::max_or_nan_fold(
    strikes
      .iter()
      .zip(market.iter())
      .map(|(&k, &px)| (model.price_call(s0, k, r, q, tau) - px).abs()),
  );
  println!("  max re-pricing error = {max_repricing_err:.2e}");
  assert!(max_repricing_err < 1e-3);
}

/// Heston recovered from a synthetic multi-maturity IV surface: one maturity identifies it weakly
/// (several `(v0, κ, θ, σ, ρ)` fit one smile), and the term structure pins down κ and θ.
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
  let model_true = HestonFourier::new(true_params);

  let strikes = [80.0_f64, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];
  let maturities = [0.25_f64, 0.5, 1.0, 2.0];

  let quotes: Vec<OptionQuote> = maturities
    .iter()
    .flat_map(|&tau| strikes.iter().map(move |&k| (k, tau)))
    .map(|(k, tau)| {
      OptionQuote::new(
        k,
        tau,
        model_true.price_call(s0, k, r, q, tau),
        OptionType::Call,
      )
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
  let calibrator = HestonCalibrator::new(s0, r, q, quotes.clone())
    .with_start(initial)
    .with_options(CalibrationOptions::default().with_record_history(true));

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
    "  RMSE = {:.4e}, converged = {}, {} accepted iterates recorded",
    res.rmse(),
    res.converged(),
    res.history().len()
  );

  // ToModel → ModelPricer round-trip on the full surface.
  let model = res.to_model();
  let max_err = f64::max_or_nan_fold(
    quotes
      .iter()
      .map(|quote| (model.price_call(s0, quote.strike, r, q, quote.tau) - quote.price).abs()),
  );
  println!("  max re-pricing error across surface = {max_err:.2e}");
}
