//! What every device-law case needs: the device under test, the terminal
//! statistics the comparisons are made on, and the agreement predicates.

use ndarray::Array1;
use stochastic_rs_stochastic::correlation::heston_stoch_corr::HestonStochCorrPaths;
use stochastic_rs_stochastic::interest::hjm::HjmPaths;
use stochastic_rs_stochastic::process::CompoundPaths;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::double_heston::DoubleHestonPaths;
use stochastic_rs_stochastic::volatility::heston2d::Heston2DPaths;

/// The device the cases run on. CUDA when the crate is built for it, Metal
/// otherwise; the whole binary is gated on one of the two being present.
#[cfg(feature = "cuda")]
pub(crate) type Device = stochastic_rs_stochastic::device::Cuda;
#[cfg(all(metal_backend, not(feature = "cuda")))]
pub(crate) type Device = stochastic_rs_stochastic::device::Metal;

/// Paths per comparison. Large enough that a terminal mean is stable to the
/// tolerances below, small enough that the whole file stays quick.
pub(crate) const M: usize = 4_000;

/// The mean of the paths' last point.
pub(crate) fn terminal_mean(paths: &[Array1<f32>]) -> f64 {
  let last = paths[0].len() - 1;
  paths.iter().map(|p| p[last] as f64).sum::<f64>() / paths.len() as f64
}

/// The standard deviation of the paths' last point: the statistic for a process that reverts to
/// zero, where a relative error on the mean is a ratio of two numbers near zero.
pub(crate) fn terminal_std(paths: &[Array1<f32>]) -> f64 {
  let last = paths[0].len() - 1;
  let mean = terminal_mean(paths);
  let var = paths
    .iter()
    .map(|p| (p[last] as f64 - mean).powi(2))
    .sum::<f64>()
    / paths.len() as f64;
  var.sqrt()
}

/// Host and device agree to within `tol` relative error.
pub(crate) fn agrees(host: f64, device: f64, tol: f64, what: &str) {
  assert!(
    (host / device - 1.0).abs() < tol,
    "{what}: host {host}, device {device}"
  );
}

/// Host and device agree on a terminal mean within five standard errors of the two samples: a
/// relative tolerance says nothing for a mean near zero, as [`spreads_agree`] argues.
pub(crate) fn means_agree(host: &[Array1<f32>], device: &[Array1<f32>], what: &str) {
  let last = |paths: &[Array1<f32>]| {
    let last = paths[0].len() - 1;
    paths.iter().map(|p| p[last] as f64).collect::<Vec<_>>()
  };
  means_agree_values(&last(host), &last(device), what);
}

/// [`means_agree`] on values a case has already taken from its paths.
pub(crate) fn means_agree_values(host: &[f64], device: &[f64], what: &str) {
  let stats = |values: &[f64]| {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
  };
  let (hm, he) = stats(host);
  let (dm, de) = stats(device);
  let band = 5.0 * (he * he + de * de).sqrt();
  assert!(
    (hm - dm).abs() < band,
    "{what}: host {hm}, device {dm} (band {band}, five standard errors of the estimate)"
  );
}

/// Host and device agree on a terminal spread within five standard errors, from the samples'
/// own kurtoses: a spread's standard error is `σ√((κ−1)/4M)`, ±3.4 % at `κ = 20`, `M = 4000`.
pub(crate) fn spreads_agree(host: &[Array1<f32>], device: &[Array1<f32>], what: &str) {
  let noise = |paths: &[Array1<f32>]| {
    let last = paths[0].len() - 1;
    let values: Vec<f64> = paths.iter().map(|p| p[last] as f64).collect();
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let m2 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let m4 = values.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / n;
    let kurtosis = m4 / (m2 * m2);
    (m2.sqrt(), m2.sqrt() * ((kurtosis - 1.0) / (4.0 * n)).sqrt())
  };
  let (hs, he) = noise(host);
  let (ds, de) = noise(device);
  let band = 5.0 * (he * he + de * de).sqrt();
  assert!(
    (hs - ds).abs() < band,
    "{what}: host {hs}, device {ds} (band {band}, five standard errors of the estimate)"
  );
}

/// Every point of every path is finite: the first thing a wrong kernel body
/// breaks, and the one check that costs nothing to make everywhere.
pub(crate) fn all_finite(paths: &[Array1<f32>], what: &str) {
  assert!(
    paths.iter().all(|p| p.iter().all(|v| v.is_finite())),
    "{what}: a path left the reals"
  );
}

/// Every point of every path lies in `[lo, hi]`, the boundary a clamping or
/// truncating family promises.
pub(crate) fn within(paths: &[Array1<f32>], lo: f32, hi: f32, what: &str) {
  assert!(
    paths.iter().all(|p| p.iter().all(|&v| v >= lo && v <= hi)),
    "{what}: a device path left [{lo}, {hi}]"
  );
}

/// `sample_map_view` and `sample_map` return the same numbers: sparing the device batch the
/// owning form's row copy changes nothing.
pub(crate) fn map_forms_agree<P>(process: impl Fn() -> P, what: &str)
where
  P: ProcessExt<f32, Output = Array1<f32>>,
{
  let owned = process().sample_map(M, |path| path[path.len() - 1]);
  let viewed = process().sample_map_view(M, |path| path[path.len() - 1]);
  assert_eq!(
    owned, viewed,
    "{what}: sample_map and sample_map_view disagree"
  );
}

/// A named multi-slot output as the array of its rows, in slot order.
pub(crate) trait Slots {
  type Rows;

  fn rows(self) -> Self::Rows;
}

/// Each path's rows in slot order, for statistics taken slot by slot.
pub(crate) fn slots<P: Slots>(paths: Vec<P>) -> Vec<P::Rows> {
  paths.into_iter().map(P::rows).collect()
}

impl Slots for CompoundPaths<f32> {
  type Rows = [Array1<f32>; 3];

  fn rows(self) -> Self::Rows {
    [self.arrivals, self.cumulative, self.jumps]
  }
}

impl Slots for HjmPaths<f32> {
  type Rows = [Array1<f32>; 3];

  fn rows(self) -> Self::Rows {
    [self.r, self.p, self.f]
  }
}

impl Slots for DoubleHestonPaths<f32> {
  type Rows = [Array1<f32>; 3];

  fn rows(self) -> Self::Rows {
    [self.s, self.v1, self.v2]
  }
}

impl Slots for HestonStochCorrPaths<f32> {
  type Rows = [Array1<f32>; 3];

  fn rows(self) -> Self::Rows {
    [self.s, self.v, self.rho]
  }
}

impl Slots for Heston2DPaths<f32> {
  type Rows = [Array1<f32>; 4];

  fn rows(self) -> Self::Rows {
    [self.x1, self.v1, self.x2, self.v2]
  }
}
