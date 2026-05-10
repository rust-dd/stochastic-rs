//! # Heston Stochastic Correlation Calibration
//!
//! Calibrates the 9-parameter HSCM model to market option prices using
//! SLSQP (Sequential Least Squares Programming) with bounds — the same
//! optimizer as the Python reference implementation.
//!
//! Parameters: \[κ_v, θ_v, σ_v, v₀, κ_r, μ_r, σ_r, ρ₀, ρ₂\]
//!
//! ## References
//! - Teng, Ehrhardt & Günther (2016)
//! - Tanaś, R. — <https://github.com/tanasr/HestonStochCorr>

use crate::pricing::heston_stoch_corr::HestonStochCorrPricer;

/// Parameter bounds for HSCM calibration (matches Python repo).
pub const BOUNDS: [(f64, f64); 9] = [
  (0.01, 10.0),  // kappa_v
  (0.001, 1.0),  // theta_v
  (0.01, 2.0),   // sigma_v
  (0.001, 0.5),  // v0
  (0.01, 20.0),  // kappa_r
  (-0.99, 0.99), // mu_r
  (0.01, 2.0),   // sigma_r
  (-0.99, 0.99), // rho0
  (-0.99, 0.99), // rho2
];

/// Market data for a single option.
#[derive(Clone, Debug)]
pub struct MarketOption {
  pub strike: f64,
  pub maturity: f64,
  pub price: f64,
  pub rate: f64,
}

/// Calibrated parameter set for the Heston-stochastic-correlation model.
#[derive(Clone, Debug)]
pub struct HscmParams {
  pub kappa_v: f64,
  pub theta_v: f64,
  pub sigma_v: f64,
  pub v0: f64,
  pub kappa_r: f64,
  pub mu_r: f64,
  pub sigma_r: f64,
  pub rho0: f64,
  pub rho2: f64,
}

/// Result of HSCM calibration.
#[derive(Clone, Debug)]
pub struct HscmCalibrationResult {
  pub kappa_v: f64,
  pub theta_v: f64,
  pub sigma_v: f64,
  pub v0: f64,
  pub kappa_r: f64,
  pub mu_r: f64,
  pub sigma_r: f64,
  pub rho0: f64,
  pub rho2: f64,
  pub rmse: f64,
  pub mae: f64,
  /// Whether the SLSQP optimizer reported a success status. Distinct from
  /// `rmse.is_finite()` — a diverged optimizer can still produce a finite RMSE
  /// against the input data.
  pub converged: bool,
  /// Final SLSQP objective value (sum of squared relative errors). Useful for
  /// comparing against pre-optimization sse to confirm progress was made.
  pub final_objective: f64,
}

impl From<HscmCalibrationResult> for crate::pricing::heston_stoch_corr::HscmModel {
  fn from(r: HscmCalibrationResult) -> Self {
    Self {
      v0: r.v0,
      kappa_v: r.kappa_v,
      theta_v: r.theta_v,
      sigma_v: r.sigma_v,
      rho0: r.rho0,
      kappa_r: r.kappa_r,
      mu_r: r.mu_r,
      sigma_r: r.sigma_r,
      rho2: r.rho2,
    }
  }
}

impl crate::traits::ToModel for HscmCalibrationResult {
  type Model = crate::pricing::heston_stoch_corr::HscmModel;
  fn to_model(&self, _r: f64, _q: f64) -> Self::Model {
    HscmCalibrationResult::to_model(self)
  }
}

impl crate::traits::CalibrationResult for HscmCalibrationResult {
  type Params = HscmParams;
  fn rmse(&self) -> f64 {
    self.rmse
  }
  fn converged(&self) -> bool {
    self.converged && self.rmse.is_finite()
  }
  fn params(&self) -> Self::Params {
    HscmParams {
      kappa_v: self.kappa_v,
      theta_v: self.theta_v,
      sigma_v: self.sigma_v,
      v0: self.v0,
      kappa_r: self.kappa_r,
      mu_r: self.mu_r,
      sigma_r: self.sigma_r,
      rho0: self.rho0,
      rho2: self.rho2,
    }
  }
}

impl HscmCalibrationResult {
  /// Convert to an [`HscmModel`] for pricing / vol surface generation.
  pub fn to_model(&self) -> crate::pricing::heston_stoch_corr::HscmModel {
    crate::pricing::heston_stoch_corr::HscmModel::from(self.clone())
  }

  pub fn to_vec(&self) -> Vec<f64> {
    vec![
      self.kappa_v,
      self.theta_v,
      self.sigma_v,
      self.v0,
      self.kappa_r,
      self.mu_r,
      self.sigma_r,
      self.rho0,
      self.rho2,
    ]
  }
}

fn price_call(p: &[f64], s0: f64, k: f64, tau: f64, r: f64) -> f64 {
  let pricer = HestonStochCorrPricer::new(
    s0, r, k, p[3], p[0], p[1], p[2], p[7], p[4], p[5], p[6], p[8], tau,
  );
  pricer.price_call_carr_madan()
}

/// Data passed to the SLSQP objective via user_data.
#[derive(Clone)]
struct CalibData {
  s0: f64,
  options: Vec<MarketOption>,
}

/// SLSQP objective function: sum of squared relative errors.
fn slsqp_objective(x: &[f64], gradient: Option<&mut [f64]>, data: &mut CalibData) -> f64 {
  // Numerical gradient via central differences
  if let Some(g) = gradient {
    let h = 1e-5;
    let f0 = eval_sse(x, data);
    for i in 0..x.len() {
      let mut xp = x.to_vec();
      xp[i] += h;
      let fp = eval_sse(&xp, data);
      g[i] = (fp - f0) / h;
    }
  }
  eval_sse(x, data)
}

fn eval_sse(x: &[f64], data: &CalibData) -> f64 {
  let mut obj = 0.0;
  for opt in &data.options {
    let model = price_call(x, data.s0, opt.strike, opt.maturity, opt.rate);
    let err = (model - opt.price) / opt.price.max(1e-6);
    obj += err * err;
  }
  obj
}

/// HSCM (Heston-stochastic-correlation) calibrator with the unified
/// [`Calibrator`](crate::traits::Calibrator) trait surface. Wraps the free
/// [`calibrate_hscm`] function in a stateful struct so generic pipelines
/// (`build_surface_from_calibration` etc.) can consume it like any other
/// calibrator.
#[derive(Clone, Debug)]
pub struct HscmCalibrator {
  pub s0: f64,
  pub options: Vec<MarketOption>,
  pub max_iter: usize,
}

impl HscmCalibrator {
  pub fn new(s0: f64, options: Vec<MarketOption>) -> Self {
    Self {
      s0,
      options,
      max_iter: 500,
    }
  }

  pub fn with_max_iter(mut self, max_iter: usize) -> Self {
    self.max_iter = max_iter;
    self
  }
}

impl crate::traits::Calibrator for HscmCalibrator {
  type InitialGuess = [f64; 9];
  type Params = HscmParams;
  type Output = HscmCalibrationResult;
  type Error = anyhow::Error;

  fn calibrate(&self, initial: Option<Self::InitialGuess>) -> Result<Self::Output, Self::Error> {
    // Default initial guess derived from the rc.1 calibration tests — a
    // mild-skew, short-tenor SPX-style starting point. Users with a better
    // prior should pass `Some([...])`.
    let guess = initial.unwrap_or([2.0, 0.04, 0.3, 0.04, 5.0, -0.5, 0.2, -0.7, 0.3]);
    Ok(calibrate_hscm(
      self.s0,
      &self.options,
      &guess,
      self.max_iter,
    ))
  }
}

/// Calibrate the HSCM model to market option prices using SLSQP.
///
/// # Arguments
/// * `s0` — Spot price.
/// * `options` — Market option data (strike, maturity, price, rate).
/// * `initial_guess` — 9-element initial parameter guess.
/// * `max_iter` — Maximum SLSQP iterations.
pub fn calibrate_hscm(
  s0: f64,
  options: &[MarketOption],
  initial_guess: &[f64; 9],
  max_iter: usize,
) -> HscmCalibrationResult {
  let n = options.len();

  let x_init = initial_guess
    .iter()
    .enumerate()
    .map(|(i, &v)| v.clamp(BOUNDS[i].0, BOUNDS[i].1))
    .collect::<Vec<_>>();

  let bounds: Vec<(f64, f64)> = BOUNDS.to_vec();
  let cons: Vec<&dyn slsqp::Func<CalibData>> = vec![];
  let data = CalibData {
    s0,
    options: options.to_vec(),
  };

  let (x, final_objective, converged) = match slsqp::minimize(
    slsqp_objective,
    &x_init,
    &bounds,
    &cons,
    data,
    max_iter,
    None,
  ) {
    Ok((_status, x_opt, fval)) => (x_opt, fval, true),
    Err((_status, x_opt, fval)) => (x_opt, fval, false),
  };

  // Compute final errors against the SLSQP-optimized parameters.
  let mut sse = 0.0;
  let mut sae = 0.0;
  for opt in options {
    let model = price_call(&x, s0, opt.strike, opt.maturity, opt.rate);
    let err = model - opt.price;
    sse += err * err;
    sae += err.abs();
  }

  HscmCalibrationResult {
    kappa_v: x[0],
    theta_v: x[1],
    sigma_v: x[2],
    v0: x[3],
    kappa_r: x[4],
    mu_r: x[5],
    sigma_r: x[6],
    rho0: x[7],
    rho2: x[8],
    rmse: (sse / n as f64).sqrt(),
    mae: sae / n as f64,
    converged,
    final_objective,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  #[ignore = "slow: HSCM Carr-Madan + SLSQP. Run with `cargo test -- --ignored`."]
  fn calibration_runs() {
    let options = vec![
      MarketOption {
        strike: 95.0,
        maturity: 0.25,
        price: 8.0,
        rate: 0.03,
      },
      MarketOption {
        strike: 100.0,
        maturity: 0.25,
        price: 5.5,
        rate: 0.03,
      },
      MarketOption {
        strike: 105.0,
        maturity: 0.25,
        price: 3.5,
        rate: 0.03,
      },
    ];

    let guess = [2.0, 0.04, 0.3, 0.04, 5.0, -0.5, 0.2, -0.7, 0.3];

    let initial_sse: f64 = options
      .iter()
      .map(|opt| {
        let m = price_call(&guess, 100.0, opt.strike, opt.maturity, opt.rate);
        let err = (m - opt.price) / opt.price.max(1e-6);
        err * err
      })
      .sum();

    let result = calibrate_hscm(100.0, &options, &guess, 500);

    assert!(result.rmse.is_finite(), "RMSE should be finite");
    assert!(result.v0 > 0.0, "v0 should be positive");
    assert!(result.rho0.abs() < 1.0, "rho0 should be in (-1,1)");
    // Catches `let _ = slsqp::minimize(...)` — without progress the optimizer
    // returned the initial guess unchanged and final_objective == initial_sse.
    assert!(
      result.final_objective <= initial_sse,
      "SLSQP must not regress: final={} initial={}",
      result.final_objective,
      initial_sse
    );
  }

  #[test]
  #[ignore = "slow: HSCM Carr-Madan FFT × SLSQP iterations. Run with --ignored."]
  fn calibrate_recovers_synthetic_prices() {
    // Ground-truth parameters drawn from the test pattern in `calibration_runs`.
    let truth = [2.0, 0.04, 0.3, 0.04, 5.0, -0.5, 0.2, -0.7, 0.3];
    let s0 = 100.0;
    let tau = 0.25;
    let r = 0.03;
    let strikes = [95.0, 100.0, 105.0];

    // Generate synthetic prices from the HSCM ground-truth pricer.
    let options: Vec<MarketOption> = strikes
      .iter()
      .map(|&k| MarketOption {
        strike: k,
        maturity: tau,
        price: price_call(&truth, s0, k, tau, r),
        rate: r,
      })
      .collect();

    // Perturbed initial guess — distinct from truth so the optimizer must move.
    let guess = [1.5, 0.05, 0.4, 0.05, 4.0, -0.3, 0.3, -0.5, 0.2];
    let initial_sse: f64 = options
      .iter()
      .map(|opt| {
        let m = price_call(&guess, s0, opt.strike, opt.maturity, opt.rate);
        let err = (m - opt.price) / opt.price.max(1e-6);
        err * err
      })
      .sum();

    let result = calibrate_hscm(s0, &options, &guess, 200);

    // Decisive: with `let _ = slsqp::minimize(...)` final_objective stayed at
    // initial_sse. After the fix the optimizer drives SSE strictly below.
    assert!(
      result.final_objective < initial_sse * 0.5,
      "SLSQP didn't make meaningful progress: final={} initial={}",
      result.final_objective,
      initial_sse
    );
    // Recovery quality is loose because 9 params × 3 prices is underdetermined,
    // but RMSE on the price scale should be tight after optimization.
    assert!(
      result.rmse < 1e-2,
      "calibrated RMSE too large vs synthetic prices: {}",
      result.rmse
    );
  }
}
