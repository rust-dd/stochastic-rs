//! What every device-law case needs: the device under test, the terminal
//! statistics the comparisons are made on, and the agreement predicates.

use ndarray::Array1;

/// The device the cases run on. CUDA when the crate is built for it, Metal
/// otherwise; the whole binary is gated on one of the two being present.
#[cfg(feature = "cuda")]
pub(crate) type Device = stochastic_rs_stochastic::device::Cuda;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) type Device = stochastic_rs_stochastic::device::Metal;

/// Paths per comparison. Large enough that a terminal mean is stable to the
/// tolerances below, small enough that the whole file stays quick.
pub(crate) const M: usize = 4_000;

/// The mean of the paths' last point.
pub(crate) fn terminal_mean(paths: &[Array1<f32>]) -> f64 {
  let last = paths[0].len() - 1;
  paths.iter().map(|p| p[last] as f64).sum::<f64>() / paths.len() as f64
}

/// The standard deviation of the paths' last point. The statistic to compare
/// on when a process reverts to zero, where a relative error on the mean is
/// the ratio of two numbers that are both nearly zero.
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

/// Every point of every path is finite: the first thing a wrong kernel body
/// breaks, and the one check that costs nothing to make everywhere.
pub(crate) fn all_finite(paths: &[Array1<f32>], what: &str) {
  assert!(
    paths.iter().all(|p| p.iter().all(|v| v.is_finite())),
    "{what}: a device path left the reals"
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
