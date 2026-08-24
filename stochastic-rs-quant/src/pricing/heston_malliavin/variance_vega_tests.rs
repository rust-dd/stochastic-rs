use super::HestonInitialVarianceVegaStability;
use super::HestonMalliavinConfig;
use super::HestonMalliavinError;
use super::HestonMalliavinEstimator;
use super::HestonModel;
use super::VanillaPortfolio;
use crate::OptionType;
use crate::pricing::bsm::BSMCoc;
use crate::pricing::bsm::BSMPricer;
use crate::traits::ModelPricer;

const UNDER_SAMPLED_SEED: u64 = 7;
/// Analytic characteristic-function initial-variance vega of the fixture
/// model and payoff below.
const REFERENCE_CF_VEGA: f64 = 20.849_039;

/// Short-dated, high vol-of-vol, high initial-variance market fit for which a
/// tiny absolute bump is noise-dominated at small path counts.
fn under_sampled_model() -> HestonModel {
  HestonModel {
    s: 490.91,
    initial_variance: 0.299_494,
    kappa: 4.0,
    theta: 0.072_326,
    vol_of_vol: 2.646_148,
    rho: 0.390_534,
    risk_free_rate: 0.05,
    dividend_yield: 0.0,
    tau: 36.0 / 365.0,
  }
}

fn under_sampled_config(paths: usize, seed: u64) -> HestonMalliavinConfig {
  HestonMalliavinConfig {
    paths,
    steps: 32,
    seed,
    initial_variance_bump: 1e-4,
    ..HestonMalliavinConfig::default()
  }
}

fn out_of_the_money_put_vertical() -> VanillaPortfolio {
  VanillaPortfolio::vertical(OptionType::Put, 455.0, 420.0)
}

#[test]
fn relative_floor_fixes_an_under_sampled_negative_vega() {
  let model = under_sampled_model();
  let payoff = out_of_the_money_put_vertical();
  let robust_config = under_sampled_config(4_096, UNDER_SAMPLED_SEED);
  let robust = HestonMalliavinEstimator::new(model, robust_config)
    .unwrap()
    .estimate(&payoff)
    .unwrap();
  let mut legacy_config = robust_config;
  legacy_config.minimum_relative_initial_variance_bump = 0.0;
  let legacy = HestonMalliavinEstimator::new(model, legacy_config)
    .unwrap()
    .estimate(&payoff)
    .unwrap();

  let diagnostics = robust.initial_variance_vega_diagnostics;
  assert_eq!(diagnostics.requested_bump, 1e-4);
  assert!((diagnostics.effective_bump - 0.03 * model.initial_variance).abs() < 1e-15);
  assert_eq!(
    diagnostics.stability,
    HestonInitialVarianceVegaStability::Stable
  );
  assert!(robust.initial_variance_vega.value > 0.0);
  assert!(
    robust.initial_variance_vega.standard_error
      < 0.15 * legacy.initial_variance_vega.standard_error
  );
  assert!(
    (robust.initial_variance_vega.value - REFERENCE_CF_VEGA).abs()
      <= 4.0 * robust.initial_variance_vega.standard_error + 1.0
  );
  assert!(legacy.initial_variance_vega.value < 0.0);
  assert_eq!(
    legacy.initial_variance_vega_diagnostics.stability,
    HestonInitialVarianceVegaStability::SamplingUnresolved
  );
  assert_eq!(
    HestonMalliavinEstimator::new(model, legacy_config)
      .unwrap()
      .estimate_requiring_stable_initial_variance_vega(&payoff)
      .unwrap_err(),
    HestonMalliavinError::UnstableInitialVarianceVega
  );
  HestonMalliavinEstimator::new(model, robust_config)
    .unwrap()
    .estimate_requiring_stable_initial_variance_vega(&payoff)
    .unwrap();
}

#[test]
fn under_sampled_default_is_positive_across_seeds_and_converges_with_paths() {
  let model = under_sampled_model();
  let payoff = out_of_the_money_put_vertical();
  for seed in [UNDER_SAMPLED_SEED, 1, 5] {
    let estimate = HestonMalliavinEstimator::new(model, under_sampled_config(4_096, seed))
      .unwrap()
      .estimate(&payoff)
      .unwrap();
    assert!(estimate.initial_variance_vega.value > 0.0, "seed={seed}");
    assert!(
      estimate.initial_variance_vega.standard_error < 2.5,
      "seed={seed}, estimate={estimate:?}"
    );
    assert_ne!(
      estimate.initial_variance_vega_diagnostics.stability,
      HestonInitialVarianceVegaStability::BumpSensitive,
      "seed={seed}, estimate={estimate:?}"
    );
  }

  let low = HestonMalliavinEstimator::new(model, under_sampled_config(4_096, UNDER_SAMPLED_SEED))
    .unwrap()
    .estimate(&payoff)
    .unwrap();
  let high = HestonMalliavinEstimator::new(model, under_sampled_config(65_536, UNDER_SAMPLED_SEED))
    .unwrap()
    .estimate(&payoff)
    .unwrap();
  assert!(high.initial_variance_vega.standard_error < low.initial_variance_vega.standard_error);
  assert!(
    (high.initial_variance_vega.value - REFERENCE_CF_VEGA).abs()
      <= 4.0 * high.initial_variance_vega.standard_error + 0.5
  );
}

#[test]
fn deterministic_variance_limit_matches_analytic_bsm_v0_vega() {
  let model = HestonModel {
    s: 100.0,
    initial_variance: 0.04,
    kappa: 1.5,
    theta: 0.04,
    vol_of_vol: 0.0,
    rho: 0.35,
    risk_free_rate: 0.03,
    dividend_yield: 0.01,
    tau: 0.75,
  };
  let payoff = VanillaPortfolio::call(105.0);
  for seed in [11, 29, 47] {
    let config = HestonMalliavinConfig {
      paths: 32_768,
      steps: 8,
      seed,
      ..HestonMalliavinConfig::default()
    };
    let estimate = HestonMalliavinEstimator::new(model, config)
      .unwrap()
      .estimate(&payoff)
      .unwrap();
    let bump = estimate.initial_variance_vega_diagnostics.effective_bump;
    assert!((bump - 0.0012).abs() < 1e-15);
    let dt = model.tau / config.steps as f64;
    let discrete_loading =
      (1.0 - (1.0 - model.kappa * dt).powi(config.steps as i32)) / (model.kappa * model.tau);
    let price = |variance: f64| {
      BSMPricer::new(variance.sqrt(), BSMCoc::Merton1973).price_call(
        model.s,
        105.0,
        model.risk_free_rate,
        model.dividend_yield,
        model.tau,
      )
    };
    let reference = (price(model.initial_variance + bump * discrete_loading)
      - price(model.initial_variance - bump * discrete_loading))
      / (2.0 * bump);
    assert!(
      (estimate.initial_variance_vega.value - reference).abs()
        <= 5.0 * estimate.initial_variance_vega.standard_error + 0.5,
      "seed={seed}, reference={reference}, estimate={estimate:?}"
    );
    assert_ne!(
      estimate.initial_variance_vega_diagnostics.stability,
      HestonInitialVarianceVegaStability::BumpSensitive
    );
  }
}
