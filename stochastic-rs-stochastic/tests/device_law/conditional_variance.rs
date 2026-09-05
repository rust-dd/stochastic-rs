//! The discrete-time conditional-variance models. Their series starts at
//! `σ₀ z₀`, so the launch steps before writing its first point, and the
//! variance recursion reads back exactly one lag, which is the order the
//! engine's state slots bound. What the cases pin is the terminal variance of
//! the series, which is what the recursion is for.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::autoregressive::agrach::Agarch;
use stochastic_rs_stochastic::autoregressive::arch::Arch;
use stochastic_rs_stochastic::autoregressive::egarch::Egarch;
use stochastic_rs_stochastic::autoregressive::garch::Garch;
use stochastic_rs_stochastic::autoregressive::tgarch::GjrGarch;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::Device;
use super::common::M;
use super::common::agrees;
use super::common::all_finite;

const N: usize = 253;

/// The sample variance of the whole ensemble's last point. A
/// conditional-variance model is driftless, so this is the statistic its
/// parameters set.
fn terminal_variance(paths: &[Array1<f32>]) -> f64 {
  let last = paths[0].len() - 1;
  let mean = paths.iter().map(|p| p[last] as f64).sum::<f64>() / paths.len() as f64;
  paths
    .iter()
    .map(|p| (p[last] as f64 - mean).powi(2))
    .sum::<f64>()
    / paths.len() as f64
}

#[test]
fn arch_agrees_with_the_cpu_law() {
  let build = || Arch::<f32, _>::new(0.0002, Array1::from(vec![0.3]), N, Deterministic::new(149));
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "ARCH");
  agrees(
    terminal_variance(&build().sample_par(M)),
    terminal_variance(&device),
    0.10,
    "ARCH terminal variance",
  );
}

#[test]
fn garch_agrees_with_the_cpu_law() {
  let build = || {
    Garch::<f32, _>::new(
      0.00001,
      Array1::from(vec![0.1]),
      Array1::from(vec![0.85]),
      N,
      Deterministic::new(151),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "GARCH");
  agrees(
    terminal_variance(&build().sample_par(M)),
    terminal_variance(&device),
    0.15,
    "GARCH terminal variance",
  );
}

/// The threshold term enters only after a negative return, so a path that has
/// seen one is more volatile than one that has not; the ensemble variance
/// carries that asymmetry.
#[test]
fn gjr_garch_agrees_with_the_cpu_law() {
  let build = || {
    GjrGarch::<f32, _>::new(
      0.00001,
      Array1::from(vec![0.05]),
      Array1::from(vec![0.1]),
      Array1::from(vec![0.85]),
      N,
      Deterministic::new(157),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "GJR-GARCH");
  agrees(
    terminal_variance(&build().sample_par(M)),
    terminal_variance(&device),
    0.15,
    "GJR-GARCH terminal variance",
  );
}

/// The asymmetric GARCH takes the same threshold term under another name.
#[test]
fn asymmetric_garch_agrees_with_the_cpu_law() {
  let build = || {
    Agarch::<f32, _>::new(
      0.00001,
      Array1::from(vec![0.05]),
      Array1::from(vec![0.1]),
      Array1::from(vec![0.85]),
      N,
      Deterministic::new(163),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "asymmetric GARCH");
  agrees(
    terminal_variance(&build().sample_par(M)),
    terminal_variance(&device),
    0.15,
    "asymmetric GARCH terminal variance",
  );
}

/// EGARCH runs its variance recursion in log space and reads back the
/// previous standardised residual, which the device recovers from the state
/// rather than keeping a third series.
#[test]
fn exponential_garch_agrees_with_the_cpu_law() {
  let build = || {
    Egarch::<f32, _>::new(
      -0.2,
      Array1::from(vec![0.1]),
      Array1::from(vec![-0.05]),
      Array1::from(vec![0.95]),
      N,
      Deterministic::new(167),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "EGARCH");
  agrees(
    terminal_variance(&build().sample_par(M)),
    terminal_variance(&device),
    0.15,
    "EGARCH terminal variance",
  );
}
