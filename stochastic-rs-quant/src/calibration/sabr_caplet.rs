//! Per-expiry Sabr smile calibration for caplets / swaptions.
//!
//! $$
//! (\hat\alpha,\hat\nu,\hat\rho)=\arg\min_{\alpha,\nu,\rho}
//!   \sum_i w_i\bigl(\sigma_{\mathrm{Hagan}}(K_i,F,\tau;\alpha,\beta,\nu,\rho)
//!   -\sigma_i^{mkt}\bigr)^2
//! $$
//!
//! The Cev exponent $\beta$ is held fixed at a user-supplied value (commonly
//! 0.5 or 1.0). The three free parameters $(\alpha,\nu,\rho)$ are minimized
//! against the supplied per-strike Black-76 implied volatilities by
//! Nelder-Mead.
//!
//! Reference: P. S. Hagan, D. Kumar, A. S. Lesniewski, D. E. Woodward,
//! "Managing Smile Risk", Wilmott Magazine (2002).

use argmin::core::CostFunction;
use argmin::core::Executor;
use argmin::core::State;
use argmin::solver::neldermead::NelderMead;

use crate::pricing::sabr::hagan_implied_vol;

/// Calibrated parameter set for a Sabr caplet smile.
///
/// Parallels [`crate::calibration::SabrParams`] but is dedicated to the
/// caplet calibrator so the two pipelines can evolve independently.
#[derive(Debug, Clone, Copy)]
pub struct SabrCapletParams {
  pub alpha: f64,
  pub beta: f64,
  pub nu: f64,
  pub rho: f64,
}

/// Calibration result for a Sabr caplet smile.
#[derive(Debug, Clone)]
pub struct SabrCapletCalibrationResult {
  /// Sabr level $\alpha$.
  pub alpha: f64,
  /// Sabr Cev exponent $\beta$ (held fixed during calibration).
  pub beta: f64,
  /// Sabr volatility of volatility $\nu$.
  pub nu: f64,
  /// Sabr correlation $\rho$.
  pub rho: f64,
  /// Root-mean-square vol error across strikes.
  pub rmse: f64,
  /// Residuals $\sigma_{model}-\sigma_{mkt}$ in strike order.
  pub residuals: Vec<f64>,
  /// True when Nelder-Mead reported convergence.
  pub converged: bool,
}

impl SabrCapletCalibrationResult {
  /// Convert to a [`SabrModel`](crate::pricing::sabr::SabrModel) for pricing /
  /// vol-surface generation.
  pub fn to_model(&self) -> crate::pricing::sabr::SabrModel {
    crate::pricing::sabr::SabrModel {
      alpha: self.alpha,
      beta: self.beta,
      nu: self.nu,
      rho: self.rho,
    }
  }
}

impl crate::traits::ToModel for SabrCapletCalibrationResult {
  type Model = crate::pricing::sabr::SabrModel;
  fn to_model(&self, _r: f64, _q: f64) -> Self::Model {
    SabrCapletCalibrationResult::to_model(self)
  }
}

impl crate::traits::CalibrationResult for SabrCapletCalibrationResult {
  type Params = SabrCapletParams;
  fn rmse(&self) -> f64 {
    self.rmse
  }
  fn converged(&self) -> bool {
    self.converged
  }
  fn params(&self) -> Self::Params {
    SabrCapletParams {
      alpha: self.alpha,
      beta: self.beta,
      nu: self.nu,
      rho: self.rho,
    }
  }
}

impl crate::traits::Calibrator for SabrCapletCalibrator {
  type InitialGuess = (f64, f64, f64);
  type Params = SabrCapletParams;
  type Output = SabrCapletCalibrationResult;
  type Error = anyhow::Error;

  fn calibrate(&self, initial: Option<Self::InitialGuess>) -> Result<Self::Output, Self::Error> {
    let mut this = self.clone();
    if let Some(g) = initial {
      this.initial_guess = Some(g);
    }
    Ok(this.solve())
  }
}

/// Sabr caplet smile calibrator — fits $(\alpha,\nu,\rho)$ for a single
/// expiry.
#[derive(Debug, Clone)]
pub struct SabrCapletCalibrator {
  /// Forward rate $F$ seen by every strike on this smile.
  pub forward: f64,
  /// Expiry $\tau$ in years.
  pub expiry: f64,
  /// Fixed Cev exponent $\beta$.
  pub beta: f64,
  /// Market strikes.
  pub strikes: Vec<f64>,
  /// Market Black-76 implied vols aligned with `strikes`.
  pub market_vols: Vec<f64>,
  /// Optional per-strike weights.
  pub weights: Option<Vec<f64>>,
  /// Initial guess for $(\alpha,\nu,\rho)$.
  pub initial_guess: Option<(f64, f64, f64)>,
  /// Maximum Nelder-Mead iterations.
  pub max_iters: u64,
  /// Convergence tolerance on the simplex standard deviation.
  pub sd_tolerance: f64,
}

impl SabrCapletCalibrator {
  /// Construct a calibrator with defaults.
  pub fn new(
    forward: f64,
    expiry: f64,
    beta: f64,
    strikes: Vec<f64>,
    market_vols: Vec<f64>,
  ) -> Self {
    assert_eq!(
      strikes.len(),
      market_vols.len(),
      "strikes and market_vols must have equal length"
    );
    Self {
      forward,
      expiry,
      beta,
      strikes,
      market_vols,
      weights: None,
      initial_guess: None,
      max_iters: 600,
      sd_tolerance: 1e-10,
    }
  }

  fn solve(&self) -> SabrCapletCalibrationResult {
    let weights = self
      .weights
      .clone()
      .unwrap_or_else(|| vec![1.0; self.strikes.len()]);

    let (a0, nu0, rho0) = self.initial_guess.unwrap_or_else(|| {
      let atm = self
        .market_vols
        .iter()
        .zip(self.strikes.iter())
        .min_by(|a, b| {
          (a.1 - self.forward)
            .abs()
            .partial_cmp(&(b.1 - self.forward).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(v, _)| *v)
        .unwrap_or(0.2);
      (atm * self.forward.powf(1.0 - self.beta), 0.4, -0.2)
    });

    let problem = SabrCapletCost {
      forward: self.forward,
      expiry: self.expiry,
      beta: self.beta,
      strikes: self.strikes.clone(),
      market_vols: self.market_vols.clone(),
      weights: weights.clone(),
    };

    let simplex = vec![
      vec![a0, nu0, rho0],
      vec![a0 * 1.3, nu0, rho0],
      vec![a0, nu0 * 1.5, rho0],
      vec![a0, nu0, (rho0 + 0.2).clamp(-0.99, 0.99)],
    ];

    let mut converged = true;
    let best = match NelderMead::new(simplex.clone()).with_sd_tolerance(self.sd_tolerance) {
      Ok(solver) => match Executor::new(problem.clone(), solver)
        .configure(|s| s.max_iters(self.max_iters))
        .run()
      {
        Ok(res) => res
          .state
          .get_best_param()
          .cloned()
          .unwrap_or_else(|| simplex[0].clone()),
        Err(_) => {
          converged = false;
          simplex[0].clone()
        }
      },
      Err(_) => {
        converged = false;
        simplex[0].clone()
      }
    };

    let alpha = best[0].abs().max(1e-8);
    let nu = best[1].abs().max(1e-8);
    let rho = best[2].clamp(-0.9999, 0.9999);

    let residuals: Vec<f64> = self
      .strikes
      .iter()
      .zip(self.market_vols.iter())
      .map(|(&k, &v_mkt)| {
        let v_model = hagan_implied_vol(k, self.forward, self.expiry, alpha, self.beta, nu, rho);
        v_model - v_mkt
      })
      .collect();
    let ssr: f64 = residuals
      .iter()
      .zip(weights.iter())
      .map(|(r, w)| (w.sqrt() * r).powi(2))
      .sum();
    let rmse = (ssr / residuals.len().max(1) as f64).sqrt();

    SabrCapletCalibrationResult {
      alpha,
      beta: self.beta,
      nu,
      rho,
      rmse,
      residuals,
      converged,
    }
  }
}

#[derive(Clone)]
struct SabrCapletCost {
  forward: f64,
  expiry: f64,
  beta: f64,
  strikes: Vec<f64>,
  market_vols: Vec<f64>,
  weights: Vec<f64>,
}

impl CostFunction for SabrCapletCost {
  type Param = Vec<f64>;
  type Output = f64;

  fn cost(&self, x: &Self::Param) -> Result<f64, argmin::core::Error> {
    let alpha = x[0].abs().max(1e-8);
    let nu = x[1].abs().max(1e-8);
    let rho = x[2].clamp(-0.9999, 0.9999);
    let mut sse = 0.0;
    for ((&k, &v_mkt), &w) in self
      .strikes
      .iter()
      .zip(self.market_vols.iter())
      .zip(self.weights.iter())
    {
      let v_model = hagan_implied_vol(k, self.forward, self.expiry, alpha, self.beta, nu, rho);
      let diff = v_model - v_mkt;
      sse += w * diff * diff;
    }
    Ok(sse)
  }
}
