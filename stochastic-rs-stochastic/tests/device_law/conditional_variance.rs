//! The discrete-time conditional-variance models. Their series starts at
//! `σ₀ z₀`, so the launch steps before writing its first point, and the
//! variance recursion reads back exactly one lag, which is the order the
//! engine's state slots bound. What the cases pin is the terminal variance of
//! the series, which is what the recursion is for.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::autoregressive::agrach::Agarch;
use stochastic_rs_stochastic::autoregressive::ar::ARp;
use stochastic_rs_stochastic::autoregressive::arch::Arch;
use stochastic_rs_stochastic::autoregressive::egarch::Egarch;
use stochastic_rs_stochastic::autoregressive::garch::Garch;
use stochastic_rs_stochastic::autoregressive::ma::MAq;
use stochastic_rs_stochastic::autoregressive::tgarch::GjrGarch;
use stochastic_rs_stochastic::noise::cgns::Cgns;
use stochastic_rs_stochastic::noise::gn::Gn;
use stochastic_rs_stochastic::noise::wn::Wn;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::Device;
use super::common::M;
use super::common::agrees;
use super::common::all_finite;
use super::common::terminal_mean;

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

/// White noise has no recursion: every grid point is one draw, so the launch
/// steps before writing the first and the ensemble's variance is the one the
/// process was given.
#[test]
fn white_noise_agrees_with_the_cpu_law() {
  let build = || Wn::<f32, _>::new(N, Some(0.01), Some(0.2), Deterministic::new(173));
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "white noise");
  agrees(
    terminal_variance(&build().sample_par(M)),
    terminal_variance(&device),
    0.10,
    "white noise terminal variance",
  );
}

/// Gaussian noise is the same family at zero mean and the grid's own step
/// size, so its variance is `dt`.
#[test]
fn gaussian_noise_agrees_with_the_cpu_law() {
  let build = || Gn::<f32, _>::new(N, Some(1.0), Deterministic::new(179));
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "Gaussian noise");
  agrees(
    terminal_variance(&build().sample_par(M)),
    terminal_variance(&device),
    0.10,
    "Gaussian noise terminal variance",
  );
}

/// A correlated pair: the second row is drawn independently and correlated in
/// the step, so what the device must reproduce is that correlation.
#[test]
fn correlated_gaussian_noise_agrees_with_the_cpu_law() {
  let build = || Cgns::<f32, _>::new(-0.5, N, Some(1.0), Deterministic::new(181));
  let device = build().on::<Device>().sample_par(M);
  let host = build().sample_par(M);
  let corr = |paths: &[[Array1<f32>; 2]]| {
    let last = paths[0][0].len() - 1;
    let (mut sxy, mut sxx, mut syy) = (0.0f64, 0.0f64, 0.0f64);
    for p in paths {
      let (x, y) = (p[0][last] as f64, p[1][last] as f64);
      sxy += x * y;
      sxx += x * x;
      syy += y * y;
    }
    sxy / (sxx * syy).sqrt()
  };
  agrees(corr(&host), corr(&device), 0.10, "correlated noise");
}

/// A first-order autoregression started from a given value: the launch does
/// not step before the first point, since that point is the value itself.
#[test]
fn autoregression_agrees_with_the_cpu_law() {
  let build = || {
    ARp::<f32, _>::new(
      Array1::from(vec![0.6]),
      0.2,
      N,
      Some(Array1::from(vec![0.5])),
      Deterministic::new(191),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  assert!(
    device.iter().all(|p| (p[0] - 0.5).abs() < 1e-6),
    "a device path did not start at the given value"
  );
  agrees(
    terminal_variance(&build().sample_par(M)),
    terminal_variance(&device),
    0.10,
    "autoregression terminal variance",
  );
}

/// A first-order moving average carries its lagged innovation as state, so
/// the terminal variance is `σ²(1 + θ²)`.
#[test]
fn moving_average_agrees_with_the_cpu_law() {
  let build = || MAq::<f32, _>::new(Array1::from(vec![0.4]), 0.2, N, Deterministic::new(193));
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "moving average");
  agrees(
    terminal_variance(&build().sample_par(M)),
    terminal_variance(&device),
    0.10,
    "moving average terminal variance",
  );
  agrees(
    terminal_mean(&build().sample_par(M)) + 1.0,
    terminal_mean(&device) + 1.0,
    0.02,
    "moving average terminal mean",
  );
}
