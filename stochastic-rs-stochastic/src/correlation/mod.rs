//! Bounded stochastic correlation on (−1, 1): [`TransformedOu`] maps an OU through a bounded $f$
//! (Section 2.1), [`VanEmmerich`] is the Jacobi-type SDE (Eq. 15), [`TengScp`] Eq. 19/20.
//!
//! Reference: Teng, Ehrhardt & Günther (2016), J. Math. Industry 6:2

pub mod heston_stoch_corr;
pub mod teng;
pub mod transformed_ou;
pub mod van_emmerich;

pub use heston_stoch_corr::HestonStochCorr;
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
pub use teng::TengScp;
pub use transformed_ou::Transformation;
pub use transformed_ou::TransformedOu;
pub use van_emmerich::VanEmmerich;

/// `[dW₁, dW₂]`, each scaled by `√dt`, with `Corr(dW₁[i], dW₂[i]) = rho_path[i]` (Teng et al. 2016,
/// Eq. 43); panics if an entry lies outside `[-1, 1]`. A constant ρ samples faster through `Cgns`.
pub fn stochastic_correlated_bm<S: SeedExt>(
  rho_path: &Array1<f64>,
  t: Option<f64>,
  seed: &S,
) -> [Array1<f64>; 2] {
  for (i, &r) in rho_path.iter().enumerate() {
    assert!(
      (-1.0..=1.0).contains(&r),
      "rho_path[{i}] must satisfy `(-1.0..=1.0).contains(&rho_path[i])`, got rho_path[{i}] = {r:?}"
    );
  }
  let n = rho_path.len();
  if n == 0 {
    return [Array1::zeros(0), Array1::zeros(0)];
  }

  let horizon = t.unwrap_or(1.0);
  let sqrt_dt = (horizon / n as f64).sqrt();

  let mut normal = SimdNormal::<f64>::new(0.0, 1.0).seeded(seed);
  let mut dw2 = normal.sample_n(n);
  let dw3 = normal.sample_n(n);

  let mut dw1 = Array1::zeros(n);
  for (i, &r) in rho_path.iter().enumerate() {
    let c = (1.0 - r * r).sqrt();
    dw1[i] = (r * dw2[i] + c * dw3[i]) * sqrt_dt;
    dw2[i] *= sqrt_dt;
  }

  [dw1, dw2]
}

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  #[test]
  fn a_seeded_pair_is_a_function_of_the_seed() {
    let rho = Array1::from_elem(256, 0.3);
    let pair = |seed: u64| stochastic_correlated_bm(&rho, None, &Deterministic::new(seed));
    assert_eq!(pair(1), pair(1));
    assert_ne!(pair(1), pair(2));
  }

  /// The sample correlation's standard error is `(1 − ρ²)/√n`, a variance ratio's `√(2/n)`.
  #[test]
  fn the_increments_carry_the_correlation_and_the_step_variance() {
    let (n, rho, t) = (20_000, 0.6, 2.0);
    let [dw1, dw2] =
      stochastic_correlated_bm(&Array1::from_elem(n, rho), Some(t), &Deterministic::new(1));
    let dt = t / n as f64;
    let (m1, m2) = (dw1.mean().unwrap(), dw2.mean().unwrap());
    let moment = |f: &dyn Fn(usize) -> f64| (0..n).map(f).sum::<f64>() / (n - 1) as f64;
    let v1 = moment(&|i| (dw1[i] - m1).powi(2));
    let v2 = moment(&|i| (dw2[i] - m2).powi(2));
    let corr = moment(&|i| (dw1[i] - m1) * (dw2[i] - m2)) / (v1 * v2).sqrt();
    let root_n = (n as f64).sqrt();
    assert!(
      (corr - rho).abs() < 5.0 * (1.0 - rho * rho) / root_n,
      "corr {corr}"
    );
    for v in [v1, v2] {
      assert!(
        (v / dt - 1.0).abs() < 5.0 * 2.0_f64.sqrt() / root_n,
        "var / dt {}",
        v / dt
      );
    }
  }

  #[test]
  fn a_refused_correlation_draws_nothing() {
    let seed = Deterministic::new(5);
    let before = seed.current();
    let refused =
      std::panic::catch_unwind(|| stochastic_correlated_bm(&array![0.2, 1.5], None, &seed));
    assert!(refused.is_err());
    assert_eq!(seed.current(), before);
  }
}
