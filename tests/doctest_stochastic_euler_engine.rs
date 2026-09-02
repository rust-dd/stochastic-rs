// docs: processes#euler-engine
//! Backs the Euler-engine example on the processes page.

use stochastic_rs::stochastic::device::Cpu;
use stochastic_rs::stochastic::diffusion::gbm::Gbm;
use stochastic_rs::stochastic::euler::EulerSpec;
use stochastic_rs::stochastic::euler::sample_paths;
use stochastic_rs_core::simd_rng::Unseeded;

#[test]
fn euler_engine_prices_a_forward() {
  let gbm = Gbm::new(0.05, 0.2, 253, Some(100.0), Some(1.0), Unseeded);
  // `Cpu` today; `CubeCl` with the gpu-cuda / gpu-wgpu feature — same call.
  let paths = sample_paths::<f64, Cpu, _>(&gbm, 20_000, 7); // Array2<f64>, shape (20_000, 253)
  assert_eq!(paths.dim(), (20_000, 253));
  let terminal_mean = paths.column(252).mean().unwrap();
  assert!((terminal_mean / (100.0 * 0.05_f64.exp()) - 1.0).abs() < 0.01);

  // The stepper is exposed too: one Euler step of dX = μX dt + σX dW.
  let spec = EulerSpec::GeometricBrownian {
    mu: 0.05,
    sigma: 0.2,
  };
  let next = spec.step(100.0, 1.0 / 252.0, (1.0_f64 / 252.0).sqrt(), 0.0);
  assert!((next - 100.0 * (1.0 + 0.05 / 252.0)).abs() < 1e-12);
}
