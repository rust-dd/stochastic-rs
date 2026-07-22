//! # Calibration
//!
//! $$
//! \hat\theta=\arg\min_\theta\sum_i w_i\left(P_i^{model}(\theta)-P_i^{mkt}\right)^2
//! $$
//!
use std::sync::OnceLock;

use gauss_quad::GaussLegendre;
use nalgebra::DVector;
use stochastic_rs_distributions::FloatExt;

use crate::CalibrationLossScore;

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

#[cfg(test)]
mod quadrature_tests;

pub use bsm::BSMCalibrationResult;
pub use bsm::BSMCalibrator;
pub use bsm::BSMParams;
pub use cgmysv::CgmysvCalibrationResult;
pub use cgmysv::CgmysvCalibrator;
pub use double_heston::DoubleHestonCalibrationResult;
pub use double_heston::DoubleHestonCalibrator;
pub use double_heston::DoubleHestonParams;
pub use heston::HestonCalibrationResult;
pub use heston::HestonCalibrator;
pub use heston::HestonParams;
pub use heston_stoch_corr::HscmCalibrationResult;
pub use heston_stoch_corr::HscmParams;
pub use heston_stoch_corr::MarketOption;
pub use heston_stoch_corr::calibrate_hscm;
pub use hkde::HKDECalibrationResult;
pub use hkde::HKDECalibrator;
pub use hkde::HKDEParams;
pub use hw_swaption::HullWhiteCalibrationResult;
pub use hw_swaption::HullWhiteParams;
pub use hw_swaption::HullWhiteSwaptionCalibrator;
pub use hw_swaption::SwaptionQuote;
pub use levy::LevyCalibrationResult;
pub use levy::LevyCalibrator;
pub use levy::LevyModelType;
pub use levy::LevyParams;
pub use levy::MarketSlice;
pub use sabr::SabrCalibrationResult;
pub use sabr::SabrCalibrator;
pub use sabr::SabrParams;
pub use sabr_caplet::SabrCapletCalibrationResult;
pub use sabr_caplet::SabrCapletCalibrator;
pub use sabr_caplet::SabrCapletParams;
pub use svj::SVJCalibrationResult;
pub use svj::SVJCalibrator;
pub use svj::SVJParams;

const GL_PANEL_WIDTH: f64 = 50.0;
const GL_MAX_PANELS: usize = 256;

fn gauss_legendre_64() -> &'static GaussLegendre {
  static GL64: OnceLock<GaussLegendre> = OnceLock::new();
  GL64.get_or_init(|| GaussLegendre::new(64.try_into().unwrap()))
}

fn compensated_add<T: FloatExt>(sum: &mut T, correction: &mut T, value: T) {
  let adjusted = value - *correction;
  let next = *sum + adjusted;
  *correction = (next - *sum) - adjusted;
  *sum = next;
}

/// Integrate coupled characteristic-function terms over `[0, ∞)`.
///
/// Every fixed-width panel receives a fresh 64-point Gauss-Legendre rule, so
/// extending the effective upper bound also increases the node count without
/// reducing node density. Two consecutive panels must be negligible in every
/// component; this keeps price and gradient integrals on the same converged
/// domain and avoids stopping on a single oscillatory cancellation.
pub(crate) fn integrate_gl_to_convergence<T: FloatExt, const N: usize, F>(
  integrand: F,
  tol: T,
) -> Option<[T; N]>
where
  F: Fn(f64) -> Option<[T; N]>,
{
  debug_assert!(N > 0);
  debug_assert!(tol.is_finite() && tol > T::zero());

  let quadrature = gauss_legendre_64();
  let half_width = 0.5 * GL_PANEL_WIDTH;
  let mut total = [T::zero(); N];
  let mut total_correction = [T::zero(); N];
  let mut negligible_streak = 0usize;

  for panel_index in 0..GL_MAX_PANELS {
    let midpoint = (panel_index as f64 + 0.5) * GL_PANEL_WIDTH;
    let mut panel = [T::zero(); N];
    let mut panel_correction = [T::zero(); N];

    for (node, weight) in quadrature.nodes().zip(quadrature.weights()) {
      let values = integrand(midpoint + half_width * *node)?;
      if values.iter().any(|value| !value.is_finite()) {
        return None;
      }

      for component in 0..N {
        compensated_add(
          &mut panel[component],
          &mut panel_correction[component],
          T::from_f64_fast(half_width * *weight) * values[component],
        );
      }
    }

    for component in 0..N {
      compensated_add(
        &mut total[component],
        &mut total_correction[component],
        panel[component],
      );
    }

    let negligible =
      (0..N).all(|component| panel[component].abs() <= tol * total[component].abs().max(T::one()));
    if negligible {
      negligible_streak += 1;
      if negligible_streak == 2 {
        return Some(total);
      }
    } else {
      negligible_streak = 0;
    }
  }

  None
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
  pub fn metric_history(history: &[Self], metric: crate::LossMetric) -> Vec<f64> {
    history.iter().map(|h| h.loss_scores.get(metric)).collect()
  }
}
