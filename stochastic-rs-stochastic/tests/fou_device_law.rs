//! fOU reaches a device through the Euler engine: the fGN pipeline supplies
//! the increments and the kernel runs the recursion, so the same OU family
//! serves the fractional process. The device and the host draw different
//! streams, so what is pinned here is the law, not the path.

#![cfg(feature = "metal")]

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::device::Metal;
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
