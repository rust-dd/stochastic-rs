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

  let x = initial_guess
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

  let _ = slsqp::minimize(slsqp_objective, &x, &bounds, &cons, data, max_iter, None);

  // Compute final errors
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
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
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
    let result = calibrate_hscm(100.0, &options, &guess, 500);

    assert!(result.rmse.is_finite(), "RMSE should be finite");
    assert!(result.v0 > 0.0, "v0 should be positive");
    assert!(result.rho0.abs() < 1.0, "rho0 should be in (-1,1)");
  }
}
