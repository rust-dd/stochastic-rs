//! The scale-free tests run at unit scale: a power of two moves no bit of an answer, whatever the
//! size of the data, and an answer in the data's units moves by that power of two alone.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

mod estimators;
mod tests;

pub(crate) fn normals(seed: u64, n: usize, sd: f64) -> Vec<f64> {
  let mut z = vec![0.0; n];
  SimdNormal::<f64>::new(0.0, sd)
    .seeded(&Deterministic::new(seed))
    .fill_slice(&mut z);
  z
}

/// A sample at a decimal size and its twin `2^k` times it, an exact product: at the ends of the
/// doubles the twin is of ordinary size, so the pair pins the end's answer to the ordinary one.
pub(crate) fn twins(sample: &[f64], reach: i32) -> Vec<(Array1<f64>, Array1<f64>, f64)> {
  let ends = [
    (1e-200, 664_i32),
    (1e-170, 565),
    (1e300, -996),
    (1e-60, 199),
    (1e60, -199),
  ];
  let near = [
    (1.0, -1_i32),
    (1.0, 1),
    (1.0, -150),
    (1.0, 150),
    (1.0, -400),
    (1.0, 400),
    (3.0, 7),
  ];
  let cases = near.into_iter().chain(ends);
  cases
    .filter(|(_, k)| k.abs() <= reach)
    .map(|(size, k)| {
      let power = 2f64.powi(k);
      let at = Array1::from_iter(sample.iter().map(|v| v * size));
      let twin = at.mapv(|v| v * power);
      (at, twin, power)
    })
    .collect()
}

pub(crate) fn assert_no_bit_moves(
  sample: &[f64],
  what: &str,
  answer: impl Fn(&Array1<f64>) -> String,
) {
  for (at, twin, power) in twins(sample, 1000) {
    assert_eq!(
      answer(&at),
      answer(&twin),
      "{what} at {:e}, × {power:e}",
      at[0]
    );
  }
}
