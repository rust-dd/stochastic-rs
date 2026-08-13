// docs: distributions#poisson-discrete-count-distribution
//! Backs the Poisson example on the distributions catalog page.

use stochastic_rs::distributions::poisson::SimdPoisson;
use stochastic_rs::simd_rng::Deterministic;

#[test]
fn poisson_bulk_sample_mean() {
  let d = SimdPoisson::<u32>::new(/* lambda */ 4.0, &Deterministic::new(42));
  let mut buf = vec![0_u32; 10_000];
  d.fill_slice(&mut buf);

  let mean = buf.iter().sum::<u32>() as f64 / 10_000.0;
  assert!(
    (mean - 4.0).abs() < 0.1,
    "sample mean = {mean:.3} (expect ~4.0)"
  );
}
