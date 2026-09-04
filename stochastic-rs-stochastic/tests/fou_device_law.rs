//! The fGN-driven processes reach a device through the Euler engine: the fGN
//! pipeline supplies the increments and the kernel runs the recursion, so the
//! same families serve them as serve their Gaussian counterparts. The device
//! and the host draw different streams, so what is pinned here is the law,
//! not the path.

#![cfg(feature = "metal")]

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::device::Metal;
use stochastic_rs_stochastic::diffusion::fcir::Fcir;
use stochastic_rs_stochastic::diffusion::fgbm::Fgbm;
use stochastic_rs_stochastic::diffusion::fjacobi::FJacobi;
use stochastic_rs_stochastic::diffusion::fou::Fou;
use stochastic_rs_stochastic::traits::ProcessExt;

fn fou() -> Fou<f32, Deterministic> {
  Fou::new(
    0.7,
    2.0,
    1.0,
    0.3,
    512,
    Some(0.0),
    Some(1.0),
    Deterministic::new(9),
  )
}

fn terminal_mean(paths: &[Array1<f32>]) -> f64 {
  paths.iter().map(|p| p[511] as f64).sum::<f64>() / paths.len() as f64
}

#[test]
fn fou_on_metal_agrees_with_the_cpu_law() {
  let m = 2_000;
  let cpu = fou().sample_par(m);
  let gpu = fou().on::<Metal>().sample_par(m);

  assert_eq!(gpu.len(), m);
  assert_eq!(gpu[0].len(), 512);
  assert!(
    gpu.iter().all(|p| p.iter().all(|v| v.is_finite())),
    "the device produced a non-finite path"
  );
  assert_eq!(gpu[0][0], 0.0, "every path starts at x0");

  let (host, device) = (terminal_mean(&cpu), terminal_mean(&gpu));
  assert!(
    (host - device).abs() < 0.05,
    "terminal mean: host {host}, device {device}"
  );
}

fn fgbm() -> Fgbm<f32, Deterministic> {
  Fgbm::new(
    0.7,
    0.05,
    0.2,
    512,
    Some(100.0),
    Some(1.0),
    Deterministic::new(9),
  )
}

fn fcir() -> Fcir<f32, Deterministic> {
  Fcir::new(
    0.7,
    2.0,
    0.04,
    0.1,
    512,
    Some(0.04),
    Some(1.0),
    None,
    Deterministic::new(9),
  )
}

#[test]
fn fgbm_on_metal_agrees_with_the_cpu_law() {
  let m = 2_000;
  let cpu = terminal_mean(&fgbm().sample_par(m));
  let gpu_paths = fgbm().on::<Metal>().sample_par(m);
  let gpu = terminal_mean(&gpu_paths);
  assert!(
    gpu_paths.iter().all(|p| p.iter().all(|v| v.is_finite())),
    "the device produced a non-finite path"
  );
  assert!(
    (cpu / gpu - 1.0).abs() < 0.02,
    "terminal mean: host {cpu}, device {gpu}"
  );
}

#[test]
fn fcir_on_metal_stays_nonnegative_and_agrees() {
  let m = 2_000;
  let cpu = terminal_mean(&fcir().sample_par(m));
  let gpu_paths = fcir().on::<Metal>().sample_par(m);
  let gpu = terminal_mean(&gpu_paths);
  assert!(
    gpu_paths.iter().all(|p| p.iter().all(|v| *v >= 0.0)),
    "the reflected recursion went negative"
  );
  assert!(
    (cpu - gpu).abs() < 0.01,
    "terminal mean: host {cpu}, device {gpu}"
  );
}

fn fjacobi() -> FJacobi<f32, Deterministic> {
  FJacobi::new(
    0.7,
    0.3,
    0.6,
    0.2,
    512,
    Some(0.5),
    Some(1.0),
    Deterministic::new(9),
  )
}

#[test]
fn fjacobi_on_metal_stays_in_the_unit_interval() {
  let m = 2_000;
  let cpu = terminal_mean(&fjacobi().sample_par(m));
  let gpu_paths = fjacobi().on::<Metal>().sample_par(m);
  let gpu = terminal_mean(&gpu_paths);
  assert!(
    gpu_paths
      .iter()
      .all(|p| p.iter().all(|v| (0.0..=1.0).contains(v))),
    "the absorbing recursion left the unit interval"
  );
  assert!(
    (cpu - gpu).abs() < 0.02,
    "terminal mean: host {cpu}, device {gpu}"
  );
}
