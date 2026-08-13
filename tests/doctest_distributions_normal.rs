// docs: distributions#normal-bulk-sampling-and-closed-form-moments
//! Backs the Normal example on the distributions catalog page.

use rand_distr::Distribution;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::simd_rng::SeedExt;
use stochastic_rs::traits::DistributionExt;

#[test]
fn normal_bulk_sample_and_closed_form() {
  let seed = Deterministic::new(42);
  let d = SimdNormal::<f64>::new(/* mean */ 0.0, /* std */ 1.0, &seed);

  // Single sample, drawn from the project's own RNG (not `rand::thread_rng`).
  let mut rng = seed.rng();
  let _x: f64 = d.sample(&mut rng);

  // Bulk fill (uses internal RNG)
  let mut buf = vec![0.0_f64; 10_000];
  d.fill_slice(&mut buf);

  // Closed-form analytics
  assert!((d.mean() - 0.0).abs() < 1e-12);
  assert!((d.variance() - 1.0).abs() < 1e-12);
  let pdf = d.pdf(0.0); // 1/sqrt(2*pi) ~= 0.3989
  let cdf = d.cdf(1.96); // ~= 0.975
  assert!((pdf - 0.398_942_280_4).abs() < 1e-6);
  assert!((cdf - 0.975).abs() < 1e-3);
}
