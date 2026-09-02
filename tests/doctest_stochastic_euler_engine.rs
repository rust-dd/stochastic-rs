// docs: processes#euler-engine
//! Backs the Euler-engine example on the processes page.

use stochastic_rs::stochastic::device::Cpu;
use stochastic_rs::stochastic::diffusion::gbm::Gbm;
use stochastic_rs::stochastic::euler::EulerCoefficients;
use stochastic_rs::stochastic::euler::sample_paths;
use stochastic_rs::traits::ProcessExt;
use stochastic_rs_core::simd_rng::Unseeded;

#[test]
fn euler_engine_prices_a_forward() {
  let gbm = Gbm::new(0.05, 0.2, 253, Some(100.0), Some(1.0), Unseeded);
  // `Cpu` is the process's own sampler; `MetalNative` / `CudaNative` / `CubeCl`
  // (with the matching feature) run the Euler kernel on the device — same call.
  let paths = sample_paths::<f64, Cpu, _>(&gbm, 20_000, 7); // Array2<f64>, shape (20_000, 253)
  assert_eq!(paths.dim(), (20_000, 253));
  let terminal_mean = paths.column(252).mean().unwrap();
  assert!((terminal_mean / (100.0 * 0.05_f64.exp()) - 1.0).abs() < 0.01);

  // On the CPU the engine returns exactly what the seeded process returns.
  let own = gbm.seeded(7).sample_par(20_000);
  assert_eq!(paths.row(0).to_vec(), own[0].to_vec());
}
