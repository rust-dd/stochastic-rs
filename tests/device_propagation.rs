use stochastic_rs::prelude::*;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::device::Cpu;
use stochastic_rs::stochastic::diffusion::fou::Fou;
use stochastic_rs::stochastic::process::fbm::Fbm;

#[test]
fn fou_on_cpu_matches_plain_sample_with_same_seed() {
  let plain = Fou::new(
    0.7_f64,
    1.0,
    0.0,
    0.2,
    64,
    Some(0.0),
    Some(1.0),
    Deterministic::new(7),
  )
  .sample();
  let on_cpu = Fou::new(
    0.7_f64,
    1.0,
    0.0,
    0.2,
    64,
    Some(0.0),
    Some(1.0),
    Deterministic::new(7),
  )
  .on::<Cpu>()
  .sample();
  assert_eq!(plain, on_cpu);
}

#[test]
fn fbm_on_cpu_matches_plain_sample_with_same_seed() {
  let plain = Fbm::new(0.7_f64, 64, Some(1.0), Deterministic::new(11)).sample();
  let on_cpu = Fbm::new(0.7_f64, 64, Some(1.0), Deterministic::new(11))
    .on::<Cpu>()
    .sample();
  assert_eq!(plain, on_cpu);
}

#[test]
fn fbm_sample_par_returns_requested_path_count() {
  let m = 8;
  let paths = Fbm::new(0.7_f64, 64, Some(1.0), Deterministic::new(3))
    .on::<Cpu>()
    .sample_par(m);
  assert_eq!(paths.len(), m);
  assert!(paths.iter().all(|p| p.len() == 64));
}
