//! Symmetric stable jump sizes, which have no variance: their spread is read off the quartiles.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::process::subordinator::ctrw::Ctrw;
use stochastic_rs_stochastic::process::subordinator::ctrw::CtrwJumpLaw;
use stochastic_rs_stochastic::process::subordinator::ctrw::CtrwWaitingLaw;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::Device;
use super::common::M;
use super::common::agrees;
use super::common::all_finite;

const N: usize = 253;

fn iqr_at(paths: &[Array1<f32>], at: usize) -> f64 {
  let mut values: Vec<f64> = paths.iter().map(|p| p[at] as f64).collect();
  values.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
  values[values.len() * 3 / 4] - values[values.len() / 4]
}

/// The terminal and quarter-horizon quartiles of `paths` device and host paths agree to `tol`.
fn quartiles_agree(alpha: f32, seed: u64, paths: usize, tol: f64, what: &str) {
  let build = || {
    Ctrw::<f32, _>::new(
      CtrwWaitingLaw::Exponential { rate: 20.0 },
      CtrwJumpLaw::SymmetricStable { alpha, scale: 0.2 },
      N,
      Some(0.0),
      Some(1.0),
      Deterministic::new(seed),
    )
  };
  let device = build().on::<Device>().sample_par(paths);
  let host = build().sample_par(paths);
  all_finite(&device, what);
  agrees(
    iqr_at(&host, N - 1),
    iqr_at(&device, N - 1),
    tol,
    &format!("{what} terminal interquartile range"),
  );
  agrees(
    iqr_at(&host, N / 4),
    iqr_at(&device, N / 4),
    tol,
    &format!("{what} quarter-horizon interquartile range"),
  );
}

/// The kernel draws a cell's sum in one Chambers–Mallows–Stuck step at scale `scale · n^{1/α}`,
/// the host draws each jump.
#[test]
fn ctrw_with_stable_jumps_agrees_with_the_cpu_law() {
  quartiles_agree(1.7, 181, 3 * M, 0.06, "CTRW stable");
}

/// Below α = 1 the step's ratio over its exponential is raised to a positive power, so the
/// smallest exponential makes the largest jump; at 48 000 paths 10 % is 4.7 standard errors.
#[test]
fn ctrw_with_stable_jumps_below_alpha_one_agrees_with_the_cpu_law() {
  quartiles_agree(0.5, 197, 12 * M, 0.1, "CTRW stable at alpha 0.5");
}
