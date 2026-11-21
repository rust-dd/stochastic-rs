//! Full truncation on a device: the kernel carries a signed variance and reports its positive
//! part, so a sub-Feller mean sits at the exact CIR mean where absorption is 10 % above it.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::diffusion::bessel::Bessel;
use stochastic_rs_stochastic::diffusion::bessel::SquaredBessel;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::fcir::Fcir;
use stochastic_rs_stochastic::interest::cir_2f::Cir2F;
use stochastic_rs_stochastic::interest::cir_pp::CirPlusPlus;
use stochastic_rs_stochastic::jump::bates::Bates;
use stochastic_rs_stochastic::jump::bates::LogNormalJump;
use stochastic_rs_stochastic::params::Drift;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::double_heston::DoubleHeston;
use stochastic_rs_stochastic::volatility::heston::Euler;
use stochastic_rs_stochastic::volatility::heston::Heston;
use stochastic_rs_stochastic::volatility::heston_log::HestonLog;
use stochastic_rs_stochastic::volatility::heston2d::Heston2D;
use stochastic_rs_stochastic::volatility::hkde::Hkde;

use super::common::Device;

const KAPPA: f32 = 0.4;
const THETA: f32 = 0.2;
const SIGMA: f32 = 0.6;
const V0: f32 = 0.05;
const T: f32 = 5.0;
const RHO: f32 = -0.7;
/// A monthly grid over `T`.
const N: usize = 61;
const PATHS: usize = 12_000;
const FT: VarianceBoundary = VarianceBoundary::FullTruncation;

/// The mean of the rows' last points and their smallest entry.
fn summary<'a>(rows: impl Iterator<Item = &'a Array1<f32>>) -> (f64, f32) {
  let (mut sum, mut count, mut least) = (0.0, 0usize, f32::INFINITY);
  for row in rows {
    sum += row[row.len() - 1] as f64;
    count += 1;
    least = row.iter().fold(least, |a, &b| a.min(b));
  }
  (sum / count as f64, least)
}

/// The device's variance rows report `ṽ⁺`: never negative, at zero somewhere, and with the
/// host's mean, both within `tol` of `exact` where absorption would sit 10 % above.
fn truncated(what: &str, device: (f64, f32), host: (f64, f32), exact: f64, tol: f64) {
  assert_eq!(device.1, 0.0, "{what}: the device's smallest variance");
  assert_eq!(host.1, 0.0, "{what}: the host's smallest variance");
  for (side, mean) in [("device", device.0), ("host", host.0)] {
    assert!(
      (mean / exact - 1.0).abs() < tol,
      "{what}: the {side} mean {mean} against the exact {exact}"
    );
  }
}

fn cir_mean() -> f64 {
  (THETA + (V0 - THETA) * (-KAPPA * T).exp()) as f64
}

#[test]
fn heston_truncates_in_full_on_the_device() {
  let build = || {
    let seed = Deterministic::new(17);
    let pow = HestonPow::Sqrt;
    Heston::<f32, Euler, _>::new(
      100.0,
      V0,
      KAPPA,
      THETA,
      SIGMA,
      RHO,
      0.0,
      N,
      Some(T),
      pow,
      FT,
      seed,
    )
  };
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  let variance = |paths: &[[Array1<f32>; 2]]| summary(paths.iter().map(|p| &p[1]));
  truncated(
    "Heston",
    variance(&device),
    variance(&host),
    cir_mean(),
    0.04,
  );
}

#[test]
fn log_heston_truncates_in_full_on_the_device() {
  let build = || {
    let seed = Deterministic::new(19);
    HestonLog::<f32, _>::new(
      Drift::Mu(0.0),
      KAPPA,
      THETA,
      SIGMA,
      RHO,
      N,
      100.0,
      V0,
      Some(T),
      FT,
      seed,
    )
  };
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  let variance = |paths: &[[Array1<f32>; 2]]| summary(paths.iter().map(|p| &p[1]));
  truncated(
    "HestonLog",
    variance(&device),
    variance(&host),
    cir_mean(),
    0.04,
  );
}

#[test]
fn double_heston_truncates_both_factors_in_full_on_the_device() {
  let build = || {
    DoubleHeston::<f32, _>::new(
      100.0,
      V0,
      V0,
      KAPPA,
      THETA,
      SIGMA,
      RHO,
      KAPPA,
      THETA,
      SIGMA,
      RHO,
      0.0,
      N,
      Some(T),
      FT,
      Deterministic::new(23),
    )
  };
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  for factor in 0..2 {
    let variance =
      |paths: &[stochastic_rs_stochastic::volatility::double_heston::DoubleHestonPaths<f32>]| {
        summary(paths.iter().map(|p| [&p.v1, &p.v2][factor]))
      };
    truncated(
      "DoubleHeston",
      variance(&device),
      variance(&host),
      cir_mean(),
      0.04,
    );
  }
}

#[test]
fn two_asset_heston_truncates_both_variances_in_full_on_the_device() {
  let build = || {
    Heston2D::<f32, _>::new(
      [4.6; 2],
      [V0; 2],
      [0.0; 2],
      [THETA; 2],
      [KAPPA; 2],
      [SIGMA; 2],
      [0.0; 6],
      N,
      Some(T),
      FT,
      Deterministic::new(29),
    )
  };
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  for asset in 0..2 {
    let variance = |paths: &[stochastic_rs_stochastic::volatility::heston2d::Heston2DPaths<
      f32,
    >]| { summary(paths.iter().map(|p| [&p.v1, &p.v2][asset])) };
    truncated(
      "Heston2D",
      variance(&device),
      variance(&host),
      cir_mean(),
      0.04,
    );
  }
}

#[test]
fn bates_truncates_in_full_on_the_device_under_either_jump_family() {
  let variance = |paths: &[[Array1<f32>; 2]]| summary(paths.iter().map(|p| &p[1]));
  let (alpha, seed) = (KAPPA * THETA, || Deterministic::new(31));
  let lognormal = || {
    let law = LogNormalJump::new(-0.02, 0.05);
    Bates::new(
      Drift::Mu(0.0),
      1.0,
      0.0,
      alpha,
      KAPPA,
      SIGMA,
      RHO,
      law,
      N,
      100.0,
      V0,
      Some(T),
      FT,
      seed(),
    )
  };
  let device = lognormal().on::<Device>().sample_par(PATHS);
  let host = lognormal().sample_par(PATHS);
  truncated(
    "Bates, log-price",
    variance(&device),
    variance(&host),
    cir_mean(),
    0.04,
  );
  let product = || {
    let law = SimdNormal::<f32>::new(-0.02, 0.05);
    Bates::new(
      Drift::Mu(0.0),
      1.0,
      -0.02,
      alpha,
      KAPPA,
      SIGMA,
      RHO,
      law,
      N,
      100.0,
      V0,
      Some(T),
      FT,
      seed(),
    )
  };
  let device = product().on::<Device>().sample_par(PATHS);
  let host = product().sample_par(PATHS);
  truncated(
    "Bates, price",
    variance(&device),
    variance(&host),
    cir_mean(),
    0.04,
  );
}

#[test]
fn kou_jump_heston_truncates_in_full_on_the_device() {
  let build = || {
    Hkde::<f32, _>::new(
      0.0,
      KAPPA,
      THETA,
      SIGMA,
      RHO,
      V0,
      1.0,
      0.4,
      10.0,
      5.0,
      N,
      100.0,
      Some(T),
      FT,
      Deterministic::new(37),
    )
  };
  let device = build().on::<Device>().sample_par(PATHS);
  let host = build().sample_par(PATHS);
  let variance = |paths: &[[Array1<f32>; 2]]| summary(paths.iter().map(|p| &p[1]));
  truncated("Hkde", variance(&device), variance(&host), cir_mean(), 0.04);
}

#[test]
fn the_square_root_rates_truncate_in_full_on_the_device() {
  let cir = |seed| {
    Cir::<f32, _>::new(
      KAPPA,
      THETA,
      SIGMA,
      N,
      Some(V0),
      Some(T),
      FT,
      Deterministic::new(seed),
    )
  };
  let device = cir(41).on::<Device>().sample_par(PATHS);
  let host = cir(41).sample_par(PATHS);
  truncated(
    "Cir",
    summary(device.iter()),
    summary(host.iter()),
    cir_mean(),
    0.04,
  );

  let shifted = || {
    let phi = (|_| 0.0) as fn(f32) -> f32;
    CirPlusPlus::<f32, _>::new(
      KAPPA,
      THETA,
      SIGMA,
      phi,
      N,
      Some(V0),
      Some(T),
      FT,
      Deterministic::new(43),
    )
  };
  let device = shifted().on::<Device>().sample_par(PATHS);
  let host = shifted().sample_par(PATHS);
  truncated(
    "CirPlusPlus",
    summary(device.iter()),
    summary(host.iter()),
    cir_mean(),
    0.04,
  );

  let two_factor = || {
    let phi = (|_| 0.0) as fn(f32) -> f32;
    Cir2F::<f32, _>::new(
      cir(47),
      cir(53).with_boundary(VarianceBoundary::Reflect),
      phi,
      Deterministic::new(59),
    )
  };
  let (device, _) = summary(two_factor().on::<Device>().sample_par(PATHS).iter());
  let (host, _) = summary(two_factor().sample_par(PATHS).iter());
  assert!(
    (device / host - 1.0).abs() < 0.04,
    "Cir2F: device {device}, host {host}"
  );
  assert!(
    device / (2.0 * cir_mean()) - 1.0 > 0.05,
    "Cir2F: the reflected factor's bias is gone"
  );
}

#[test]
fn the_squared_bessel_rows_truncate_in_full_on_the_device() {
  let (delta, x0) = (0.5f32, 0.1f32);
  let exact = (x0 + delta * T) as f64;
  let squared =
    || SquaredBessel::<f32, _>::new(delta, N, Some(x0), Some(T), FT, Deterministic::new(61));
  let device = squared().on::<Device>().sample_par(PATHS);
  let host = squared().sample_par(PATHS);
  truncated(
    "SquaredBessel",
    summary(device.iter()),
    summary(host.iter()),
    exact,
    0.08,
  );

  let root = || {
    Bessel::<f32, _>::new(
      delta,
      N,
      Some(x0.sqrt()),
      Some(T),
      FT,
      Deterministic::new(67),
    )
  };
  let (device, device_least) = summary(root().on::<Device>().sample_par(PATHS).iter());
  let (host, host_least) = summary(root().sample_par(PATHS).iter());
  assert_eq!(
    (device_least, host_least),
    (0.0, 0.0),
    "Bessel: the smallest radius"
  );
  assert!(
    (device / host - 1.0).abs() < 0.04,
    "Bessel: device {device}, host {host}"
  );
}

#[test]
fn fractional_cir_truncates_in_full_on_the_device() {
  let build = || {
    Fcir::<f32, _>::new(
      0.7,
      KAPPA,
      THETA,
      SIGMA,
      N,
      Some(V0),
      Some(T),
      FT,
      Deterministic::new(71),
    )
  };
  let (device, device_least) = summary(build().on::<Device>().sample_par(PATHS).iter());
  let (host, host_least) = summary(build().sample_par(PATHS).iter());
  assert_eq!(
    (device_least, host_least),
    (0.0, 0.0),
    "fCIR: the smallest value"
  );
  assert!(
    (device / host - 1.0).abs() < 0.04,
    "fCIR: device {device}, host {host}"
  );
}
