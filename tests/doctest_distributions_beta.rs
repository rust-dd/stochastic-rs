// docs: distributions#beta-bounded-support-distribution
//! Backs the Beta example on the distributions catalog page.

use stochastic_rs::distributions::beta::SimdBeta;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::traits::DistributionExt;

#[test]
fn beta_bounded_support_and_moments() {
  let d = SimdBeta::<f64>::new(
    /* alpha */ 2.0,
    /* beta */ 5.0,
    &Deterministic::new(42),
  );
  let mut buf = vec![0.0; 1_000];
  d.fill_slice(&mut buf);
  assert!(buf.iter().all(|&x| (0.0..=1.0).contains(&x)));

  // E[X] = alpha / (alpha + beta) = 2/7
  // Var  = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1)) = 10/(49*8)
  assert!((d.mean() - 2.0 / 7.0).abs() < 1e-12);
  assert!((d.variance() - 10.0 / (49.0 * 8.0)).abs() < 1e-12);
}
