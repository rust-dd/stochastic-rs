use super::*;

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

/// Regression: confidence-style values (e.g. `0.95`) passed as
/// `cvar_alpha` must panic loudly. The rc.0 implementation accepted any
/// `alpha ∈ [0, 1]`, silently averaging nearly the whole distribution
/// when users got the convention backwards. rc.1 rejects `alpha >= 0.5`.
#[test]
#[should_panic(expected = "tail proportion")]
fn empirical_cvar_rejects_confidence_level_misuse() {
  let mut returns = vec![-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04];
  let _ = empirical_cvar(&mut returns, 0.95);
}

#[test]
fn empirical_cvar_accepts_typical_tail_proportions() {
  let mut returns: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.001).collect();
  let cvar_5pct = empirical_cvar(&mut returns.clone(), 0.05);
  let cvar_10pct = empirical_cvar(&mut returns, 0.10);
  // CVaR at 5% tail must be MORE negative (worse loss) than 10% tail.
  assert!(
    cvar_5pct >= cvar_10pct,
    "5% tail CVaR ({cvar_5pct}) must be ≥ 10% tail CVaR ({cvar_10pct}) (more loss)"
  );
}
