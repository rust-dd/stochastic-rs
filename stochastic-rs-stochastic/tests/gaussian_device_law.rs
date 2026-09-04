//! The single-state Gaussian diffusions the Euler engine serves: each is one
//! family declaration, and the device and the host draw different streams, so
//! what is pinned here is the law and the boundaries, not the path.

#![cfg(any(feature = "metal", feature = "cuda"))]

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::cev::Cev;
use stochastic_rs_stochastic::diffusion::ckls::Ckls;
use stochastic_rs_stochastic::diffusion::gbm_log::GbmLog;
use stochastic_rs_stochastic::diffusion::logistic::Logistic;
use stochastic_rs_stochastic::diffusion::radial_ou::RadialOU;
use stochastic_rs_stochastic::diffusion::three_half::ThreeHalf;
use stochastic_rs_stochastic::traits::ProcessExt;

#[cfg(feature = "cuda")]
type Device = stochastic_rs_stochastic::device::Cuda;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
type Device = stochastic_rs_stochastic::device::Metal;

fn terminal_mean(paths: &[Array1<f32>]) -> f64 {
  let last = paths[0].len() - 1;
  paths.iter().map(|p| p[last] as f64).sum::<f64>() / paths.len() as f64
}

fn agrees(host: f64, device: f64, tol: f64, what: &str) {
  assert!(
    (host / device - 1.0).abs() < tol,
    "{what}: host {host}, device {device}"
  );
}

#[test]
fn cev_agrees_with_the_cpu_law() {
  let build = || {
    Cev::<f32, _>::new(
      0.05,
      0.2,
      0.8,
      253,
      Some(100.0),
      Some(1.0),
      Deterministic::new(5),
    )
  };
  let m = 4_000;
  let device = build().on::<Device>().sample_par(m);
  assert!(device.iter().all(|p| p.iter().all(|v| v.is_finite())));
  agrees(
    terminal_mean(&build().sample_par(m)),
    terminal_mean(&device),
    0.02,
    "CEV terminal mean",
  );
}

#[test]
fn ckls_agrees_with_the_cpu_law() {
  let build = || {
    Ckls::<f32, _>::new(
      0.06,
      -1.5,
      0.3,
      0.5,
      253,
      Some(0.04),
      Some(1.0),
      Deterministic::new(5),
    )
  };
  let m = 4_000;
  let device = build().on::<Device>().sample_par(m);
  assert!(device.iter().all(|p| p.iter().all(|v| v.is_finite())));
  agrees(
    terminal_mean(&build().sample_par(m)),
    terminal_mean(&device),
    0.05,
    "CKLS terminal mean",
  );
}

#[test]
fn logistic_agrees_with_the_cpu_law() {
  let build =
    || Logistic::<f32, _>::new(0.5, 0.2, 253, Some(1.0), Some(1.0), Deterministic::new(5));
  let m = 4_000;
  let device = build().on::<Device>().sample_par(m);
  assert!(device.iter().all(|p| p.iter().all(|v| v.is_finite())));
  agrees(
    terminal_mean(&build().sample_par(m)),
    terminal_mean(&device),
    0.03,
    "logistic terminal mean",
  );
}

#[test]
fn three_half_agrees_with_the_cpu_law() {
  let build = || {
    ThreeHalf::<f32, _>::new(
      2.0,
      0.04,
      0.3,
      253,
      Some(0.04),
      Some(1.0),
      Deterministic::new(5),
    )
  };
  let m = 4_000;
  let device = build().on::<Device>().sample_par(m);
  assert!(device.iter().all(|p| p.iter().all(|v| v.is_finite())));
  agrees(
    terminal_mean(&build().sample_par(m)),
    terminal_mean(&device),
    0.05,
    "3/2 terminal mean",
  );
}

#[test]
fn log_gbm_agrees_with_the_cpu_law() {
  let build = || {
    GbmLog::<f32, _>::new(
      Some(0.05),
      None,
      None,
      None,
      0.2,
      253,
      Some(100.0),
      Some(1.0),
      Deterministic::new(5),
    )
  };
  let m = 4_000;
  let device = build().on::<Device>().sample_par(m);
  assert!(device.iter().all(|p| p.iter().all(|v| *v > 0.0)));
  agrees(
    terminal_mean(&build().sample_par(m)),
    terminal_mean(&device),
    0.02,
    "log-GBM terminal mean",
  );
}

#[test]
fn radial_ou_agrees_with_the_cpu_law() {
  let build =
    || RadialOU::<f32, _>::new(1.0, 0.3, 253, Some(1.0), Some(1.0), Deterministic::new(5));
  let m = 4_000;
  let device = build().on::<Device>().sample_par(m);
  assert!(device.iter().all(|p| p.iter().all(|v| v.is_finite())));
  agrees(
    terminal_mean(&build().sample_par(m)),
    terminal_mean(&device),
    0.05,
    "radial OU terminal mean",
  );
}
