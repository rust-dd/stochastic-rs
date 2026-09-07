//! What every device-law case needs: the device under test, the terminal
//! statistics the comparisons are made on, and the agreement predicates.

use ndarray::Array1;
use stochastic_rs_stochastic::traits::ProcessExt;

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

/// Host and device agree on a terminal mean, within five standard errors of
/// what the two samples allow.
///
/// The same argument as [`spreads_agree`]: a relative tolerance is a guess
/// about a quantity whose own noise is `σ/√M` per side, and for a mean near
/// zero — a square-root process at `v₀ = 0.04`, a rate reverting to `0.04` —
/// the ratio of two small numbers says nothing about agreement.
pub(crate) fn means_agree(host: &[Array1<f32>], device: &[Array1<f32>], what: &str) {
  let stats = |paths: &[Array1<f32>]| {
    let last = paths[0].len() - 1;
    let values: Vec<f64> = paths.iter().map(|p| p[last] as f64).collect();
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

/// Host and device agree on a terminal spread, within five standard errors
/// of what that spread's own noise allows.
///
/// A relative tolerance is the wrong instrument for a heavy-tailed statistic:
/// the standard error of a sample standard deviation is `σ√((κ−1)/4M)`, so a
/// terminal value with a kurtosis near twenty carries ±3.4 % per side at four
/// thousand paths and ±4.9 % between two independent samples. A fixed 6 %
/// band on that is one and a quarter standard errors — a coin toss dressed as
/// a check, and it is what the Cheyette case used to be. Here the band comes
/// from the two samples' own kurtoses.
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

/// `sample_map_view` and `sample_map` return the same numbers.
///
/// The view form exists to spare the device batch a second traversal — the
/// owning form copies every row out of the launch buffer before the callback
/// sees it — so what has to be pinned is that sparing it changes nothing.
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

/// `sample_reduce` returns what folding `sample_par`'s own paths would give,
/// for every mode, on whichever backend the process carries.
///
/// The device folds in the kernel and never writes the grid; the host folds
/// the grid it wrote. What must not differ is the answer, so the comparison
/// is against the *same* backend's paths — the two backends draw different
/// streams by construction, and only their laws are comparable.
pub(crate) fn reductions_match_the_paths<P>(build: impl Fn() -> P, what: &str)
where
  P: ProcessExt<f32, Output = Array1<f32>>,
{
  use stochastic_rs_stochastic::euler::Reduce;
  let paths = build().sample_par(M);
  for reduce in [Reduce::Terminal, Reduce::Max, Reduce::Min, Reduce::Sum] {
    let folded = paths
      .iter()
      .map(|p| reduce.fold_row(p.view()))
      .collect::<Vec<f32>>();
    let reduced = build().sample_reduce(M, reduce);
    assert_eq!(reduced.len(), folded.len(), "{what} {reduce:?}: length");
    let bad = reduced.iter().zip(&folded).filter(|(a, b)| a != b).count();
    assert_eq!(
      bad, 0,
      "{what} {reduce:?}: {bad} of {M} differ from the folded paths"
    );
  }
}
