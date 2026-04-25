//! # Calibration
//!
//! $$
//! \hat\theta=\arg\min_\theta\sum_i w_i\left(P_i^{model}(\theta)-P_i^{mkt}\right)^2
//! $$
//!
use std::sync::OnceLock;

use gauss_quad::GaussLegendre;
use nalgebra::DVector;

use crate::quant::CalibrationLossScore;

pub mod bsm;
pub mod cgmysv;
pub mod double_heston;
pub mod heston;
pub mod heston_stoch_corr;
pub mod hkde;
pub mod hw_swaption;
pub mod levy;
pub mod rbergomi;
pub mod sabr;
pub mod sabr_caplet;
pub mod svj;

// Re-export key calibration types for convenience.
pub use bsm::BSMCalibrationResult;
pub use bsm::BSMCalibrator;
pub use bsm::BSMParams;
pub use cgmysv::CgmysvCalibrationResult;
pub use cgmysv::CgmysvCalibrator;
pub use double_heston::DoubleHestonCalibrationResult;
pub use double_heston::DoubleHestonCalibrator;
pub use double_heston::DoubleHestonParams;
pub use heston::HestonCalibrator;
pub use heston::HestonParams;
pub use heston_stoch_corr::HscmCalibrationResult;
pub use heston_stoch_corr::MarketOption;
pub use heston_stoch_corr::calibrate_hscm;
pub use hkde::HKDECalibrationResult;
pub use hkde::HKDECalibrator;
pub use hkde::HKDEParams;
pub use hw_swaption::HullWhiteCalibrationResult;
pub use hw_swaption::HullWhiteSwaptionCalibrator;
pub use hw_swaption::SwaptionQuote;
pub use levy::LevyCalibrationResult;
pub use levy::LevyCalibrator;
pub use levy::LevyModelType;
pub use levy::MarketSlice;
pub use sabr::SabrCalibrationResult;
pub use sabr::SabrCalibrator;
pub use sabr::SabrParams;
pub use sabr_caplet::SabrCapletCalibrationResult;
pub use sabr_caplet::SabrCapletCalibrator;
pub use svj::SVJCalibrationResult;
pub use svj::SVJCalibrator;
pub use svj::SVJParams;

/// Default upper integration limit for Gil-Pelaez integrals in calibrators.
pub(crate) const GL_U_MAX: f64 = 100.0;

/// Cached 64-point Gauss-Legendre nodes and weights via `gauss_quad` crate.
pub(crate) fn gauss_legendre_64() -> (&'static [f64], &'static [f64]) {
  static GL64: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
  let (nodes, weights) = GL64.get_or_init(|| {
    let quad = GaussLegendre::new(64.try_into().unwrap());
    let nodes: Vec<f64> = quad.nodes().copied().collect();
    let weights: Vec<f64> = quad.weights().copied().collect();
    (nodes, weights)
  });
  (nodes.as_slice(), weights.as_slice())
}

/// Periodic linear extension mapping `x` into `[c, d]`.
///
/// Used to keep optimiser parameters in range without hard clipping.
pub(crate) fn periodic_map(x: f64, c: f64, d: f64) -> f64 {
  if c <= x && x <= d {
    x
  } else {
    let range = d - c;
    if range <= 0.0 {
      return c;
    }
    let n = ((x - c) / range).floor();
    let n_int = n as i64;
    if n_int % 2 == 0 {
      x - n * range
    } else {
      d + n * range - (x - c)
    }
  }
}

#[derive(Clone, Debug)]
pub struct CalibrationHistory<T> {
  /// Residual vector from calibration objective.
  pub residuals: DVector<f64>,
  pub call_put: DVector<(f64, f64)>,
  /// Model parameter set (input or calibrated output).
  pub params: T,
  /// Calibration loss metric configuration/result.
  pub loss_scores: CalibrationLossScore,
}

impl<T> CalibrationHistory<T> {
  /// Extract the history of a single loss metric across iterations.
  pub fn metric_history(history: &[Self], metric: crate::quant::LossMetric) -> Vec<f64> {
    history.iter().map(|h| h.loss_scores.get(metric)).collect()
  }
}
