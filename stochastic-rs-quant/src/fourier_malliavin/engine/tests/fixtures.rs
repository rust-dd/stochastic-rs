//! Deterministic Heston fixtures and analytical reference quantities.

use ndarray::Array1;
use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::heston::Heston;
use stochastic_rs_stochastic::volatility::heston2d::Heston2D;

use crate::traits::ProcessExt;

pub(super) const HESTON_SIGMA_V: f64 = 1.0;
pub(super) const HESTON_RHO: f64 = -0.5;

pub(super) fn heston_paths() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
  let n = 23401_usize;
  let horizon = 1.0_f64;
  let heston = Heston::new(
    Some(100.0),
    Some(0.4),
    2.0,
    0.4,
    HESTON_SIGMA_V,
    HESTON_RHO,
    0.0,
    n,
    Some(horizon),
    HestonPow::Sqrt,
    Some(false),
    Deterministic::new(42),
  );
  let [prices, variance] = heston.sample();
  let dt = horizon / (n as f64 - 1.0);
  let times = (0..n).map(|index| index as f64 * dt).collect::<Array1<_>>();
  (prices.mapv(f64::ln), variance, times)
}

pub(super) fn true_integrated_variance(variance: ArrayView1<'_, f64>, dt: f64) -> f64 {
  (0..variance.len() - 1)
    .map(|index| (variance[index] + variance[index + 1]) * 0.5 * dt)
    .sum()
}

pub(super) fn true_integrated_leverage(variance: ArrayView1<'_, f64>, dt: f64) -> f64 {
  HESTON_SIGMA_V * HESTON_RHO * true_integrated_variance(variance, dt)
}

pub(super) fn true_integrated_volvol(variance: ArrayView1<'_, f64>, dt: f64) -> f64 {
  HESTON_SIGMA_V.powi(2) * true_integrated_variance(variance, dt)
}

/// Bivariate path matching the FSDA `Heston2D.m` parameter convention.
pub(super) fn heston2d_paths() -> (
  Array1<f64>,
  Array1<f64>,
  Array1<f64>,
  Array1<f64>,
  Array1<f64>,
) {
  let n = 23401_usize;
  let horizon = 1.0_f64;
  let heston = Heston2D::<f64, _>::new(
    [Some(0.0_f64), Some(0.0_f64)],
    [Some(0.4_f64), Some(0.4_f64)],
    [0.0, 0.0],
    [0.4, 0.4],
    [2.0, 2.0],
    [1.0, 1.0],
    [0.5, -0.5, 0.0, 0.0, -0.5, 0.5],
    n,
    Some(horizon),
    Some(false),
    Deterministic::new(42),
  );
  let [x1, v1, x2, v2] = heston.sample();
  let dt = horizon / (n as f64 - 1.0);
  let times = (0..n).map(|index| index as f64 * dt).collect::<Array1<_>>();
  (x1, x2, times, v1, v2)
}
