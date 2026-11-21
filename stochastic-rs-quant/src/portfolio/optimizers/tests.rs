use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use super::*;
use crate::risk::PnlOrLoss;
use crate::risk::historical_es;

#[test]
fn markowitz_long_only_weights_sum_to_one() {
  let mu = vec![0.08, 0.1, 0.12];
  let cov = vec![
    vec![0.04, 0.01, 0.0],
    vec![0.01, 0.09, 0.02],
    vec![0.0, 0.02, 0.16],
  ];

  let result = optimize_with_method(
    OptimizerMethod::Markowitz,
    &mu,
    &cov,
    None,
    None,
    0.1,
    0.02,
    0.05,
    false,
    &OptimizerConfig::default(),
  );

  let sum_w: f64 = result.weights.iter().sum();
  assert!((sum_w - 1.0).abs() < 1e-6);
}

#[test]
fn optimizer_handles_empty_inputs() {
  let result = optimize_with_method(
    OptimizerMethod::Markowitz,
    &[],
    &[],
    None,
    None,
    0.1,
    0.0,
    0.05,
    false,
    &OptimizerConfig::default(),
  );

  assert!(result.weights.is_empty());
  assert_eq!(result.expected_return, 0.0);
  assert_eq!(result.volatility, 0.0);
}

/// A confidence level such as `0.95` passed as the tail proportion `cvar_alpha` panics.
#[test]
#[should_panic(
  expected = "alpha must satisfy `alpha > 0.0 && alpha < 0.5`, a tail proportion, not a confidence level, got alpha = 0.95"
)]
fn empirical_cvar_rejects_confidence_level_misuse() {
  let mut returns = vec![-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04];
  let _ = empirical_cvar(&mut returns, 0.95);
}

#[test]
fn empirical_cvar_accepts_typical_tail_proportions() {
  let mut returns: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.001).collect();
  let cvar_5pct = empirical_cvar(&mut returns.clone(), 0.05).unwrap();
  let cvar_10pct = empirical_cvar(&mut returns, 0.10).unwrap();
  // CVaR at 5% tail must be MORE negative (worse loss) than 10% tail.
  assert!(
    cvar_5pct >= cvar_10pct,
    "5% tail CVaR ({cvar_5pct}) must be ≥ 10% tail CVaR ({cvar_10pct}) (more loss)"
  );
}

/// The worst `nα = 1.5` of thirty returns: all of `−10` and half of `−2`.
#[test]
fn empirical_cvar_is_the_tail_mean_of_the_losses() {
  let mut returns = vec![0.0; 28];
  returns.extend([-10.0, -2.0]);
  let cvar = empirical_cvar(&mut returns, 0.05).unwrap();
  assert!(
    (cvar - 22.0 / 3.0).abs() <= 4.0 * f64::EPSILON * 22.0 / 3.0,
    "{cvar:?}"
  );
}

#[test]
fn empirical_cvar_is_historical_es_at_the_complementary_confidence() {
  for (n, alpha) in [(30, 0.05), (63, 0.05), (250, 0.025), (101, 0.1), (7, 0.3)] {
    let normal = SimdNormal::<f64>::new(0.0, 0.01);
    let mut returns = normal
      .seeded(&Deterministic::new(n as u64))
      .sample_n(n)
      .to_vec();
    let es = historical_es(ArrayView1::from(&returns), 1.0 - alpha, PnlOrLoss::Pnl).unwrap();
    let cvar = empirical_cvar(&mut returns, alpha).unwrap();
    assert!(
      (cvar - es).abs() <= 1e-13 * es.abs(),
      "n {n}, alpha {alpha}: {cvar:?}, {es:?}"
    );
  }
}
