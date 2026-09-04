// docs: processes#euler-engine
//! Backs the Euler-engine example on the processes page.

use stochastic_rs::stochastic::device::Cpu;
use stochastic_rs::stochastic::diffusion::gbm::Gbm;
use stochastic_rs::traits::ProcessExt;
use stochastic_rs_core::simd_rng::Deterministic;

#[test]
fn euler_engine_prices_a_forward() {
  let gbm = Gbm::new(
    0.05,
    0.2,
    253,
    Some(100.0),
    Some(1.0),
    Deterministic::new(7),
  );
  // The backend is a type parameter of the process: `Cpu` (the default) is the
  // process's own SIMD sampler; `.on::<Metal>()`, `.on::<Cuda>()` or
  // `.on::<CubeCl>()` (with the matching feature) run the Euler kernel on the
  // device through the very same `sample_par` call.
  let paths = gbm.clone().on::<Cpu>().sample_par(20_000); // Vec<Array1<f64>>, 20_000 × 253
  assert_eq!(paths.len(), 20_000);
  let terminal_mean = paths.iter().map(|p| p[252]).sum::<f64>() / 20_000.0;
  assert!((terminal_mean / (100.0 * 0.05_f64.exp()) - 1.0).abs() < 0.01);

  // `on::<Cpu>()` changes nothing: it is exactly what the process returns.
  assert_eq!(paths[0].to_vec(), gbm.sample_par(20_000)[0].to_vec());
}
