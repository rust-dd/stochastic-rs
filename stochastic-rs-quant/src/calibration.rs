//! Calibrators fitting model parameters to market quotes by weighted least squares.
//!
//! $$
//! \hat\theta=\arg\min_\theta\sum_i w_i\left(P_i^{model}(\theta)-P_i^{mkt}\right)^2
//! $$
//!
use std::sync::OnceLock;

use gauss_quad::GaussLegendre;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use stochastic_rs_distributions::RealExt;

pub mod bsm;
mod calibrated;
pub mod cgmysv;
pub mod double_heston;
mod feller_share;
pub mod heston;
pub mod heston_slv;
pub mod heston_stoch_corr;
pub mod hkde;
pub mod hw_swaption;
#[doc(hidden)]
pub mod least_squares;
pub mod levy;
mod nelder_mead;
pub(crate) mod quote;
pub mod rbergomi;
pub mod regularization;
mod reseat;
pub mod sabr;
pub mod sabr_caplet;
pub(crate) mod stencil;
pub mod svj;
pub mod tree_swaption;

#[cfg(test)]
mod quadrature_tests;

pub use bsm::BsmCalibrator;
pub use bsm::BsmParams;
pub use calibrated::Calibrated;
pub use calibrated::CalibrationHistory;
pub use calibrated::CalibrationOptions;
pub(crate) use calibrated::Knob;
pub use cgmysv::CgmysvCalibrator;
pub use double_heston::DoubleHestonCalibrator;
pub use double_heston::DoubleHestonParams;
pub use heston::HestonCalibrator;
pub use heston::HestonMleSeed;
pub use heston_stoch_corr::HscmCalibrator;
pub use heston_stoch_corr::HscmParams;
pub use hkde::HkdeCalibrator;
pub use hw_swaption::HullWhiteParams;
pub use hw_swaption::HullWhiteSwaptionCalibrator;
pub use hw_swaption::SwaptionQuote;
pub use levy::LevyCalibrator;
pub use levy::LevyModelType;
pub use levy::LevyParams;
pub(crate) use nelder_mead::nelder_mead_fit;
pub(crate) use nelder_mead::run_nelder_mead;
pub(crate) use nelder_mead::start_vertex;
pub use regularization::Regularization;
pub use sabr::SabrCalibrator;
pub use sabr_caplet::SabrCapletCalibrator;
pub use sabr_caplet::SabrCapletParams;
pub use svj::SvjCalibrator;
pub use tree_swaption::BlackKarasinskiParams;
pub use tree_swaption::BlackKarasinskiSwaptionCalibrator;
pub use tree_swaption::G2ppParams;
pub use tree_swaption::G2ppSwaptionCalibrator;

const GL_PANEL_WIDTH: f64 = 50.0;
const GL_MAX_PANELS: usize = 256;

fn gauss_legendre_64() -> &'static GaussLegendre {
  static GL64: OnceLock<GaussLegendre> = OnceLock::new();
  GL64.get_or_init(|| GaussLegendre::new(64.try_into().unwrap()))
}

fn compensated_add<T: RealExt>(sum: &mut T, correction: &mut T, value: T) {
  let adjusted = value - *correction;
  let next = *sum + adjusted;
  *correction = (next - *sum) - adjusted;
  *sum = next;
}

/// Coupled chf terms over `[0, ∞)` by 64-point Gauss–Legendre panels until two in a row are
/// negligible in every component: price and gradient share a domain, no cancellation stops it.
pub(crate) fn integrate_gl_to_convergence<T: RealExt, const N: usize, F>(
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

/// A finite negative model price clipped to 0; NaN and ±∞ pass through, so the optimiser rejects
/// the trial instead of reading a price.
pub(crate) fn clip_negative(price: f64) -> f64 {
  if price.is_finite() {
    price.max(0.0)
  } else {
    price
  }
}

/// `d periodic_map / dx`: −1 where `x` lies an odd number of ranges outside `[c, d]`.
pub(crate) fn periodic_slope(x: f64, c: f64, d: f64) -> f64 {
  let reflections = ((x - c) / (d - c)).floor() as i64;
  if (c <= x && x <= d) || reflections % 2 == 0 {
    1.0
  } else {
    -1.0
  }
}

/// `periodic_slope` of each coordinate against its box.
pub(crate) fn periodic_slopes(x: ArrayView1<f64>, bounds: &[(f64, f64)]) -> Array1<f64> {
  let slopes = x.iter().zip(bounds);
  slopes
    .map(|(&x, &(c, d))| periodic_slope(x, c, d))
    .collect()
}

/// A Jacobian in projected parameters times `dθⱼ/dxⱼ`, column by column.
pub(crate) fn scale_columns(mut jacobian: Array2<f64>, slopes: ArrayView1<f64>) -> Array2<f64> {
  for (mut column, slope) in jacobian.columns_mut().into_iter().zip(slopes) {
    column *= *slope;
  }
  jacobian
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

#[cfg(test)]
mod tests {
  use super::clip_negative;

  #[test]
  fn a_finite_negative_price_clips_to_zero_and_a_non_finite_one_passes() {
    assert_eq!(clip_negative(-1e-12), 0.0);
    assert_eq!(clip_negative(3.5), 3.5);
    assert!(clip_negative(f64::NAN).is_nan());
    assert_eq!(clip_negative(f64::INFINITY), f64::INFINITY);
    assert_eq!(clip_negative(f64::NEG_INFINITY), f64::NEG_INFINITY);
  }
}
