//! Base-path-only Malliavin estimation of Heston price and spot Greeks.

use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::mc::McEstimate;

use super::TerminalPayoff;
use super::estimator::HestonMalliavinConfig;
use super::estimator::HestonModel;
use super::non_finite_payoff;
use super::simulation::simulate_path;
use super::statistics::OnlineCovariance;
use crate::QuantError;

/// Number of jointly estimated price and spot-Greek observables.
pub const HESTON_MALLIAVIN_SPOT_OBSERVABLES: usize = 3;

/// Simulation provenance for a spot-only Malliavin estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HestonMalliavinSpotProvenance {
  /// One base Heston path per raw antithetic path and no parameter-bumped paths.
  BaseAntitheticPaths,
}

/// Joint price, spot delta, and spot gamma estimate without bumped Heston paths.
#[derive(Debug, Clone, PartialEq)]
pub struct HestonMalliavinSpotEstimate {
  /// Discounted option value.
  pub price: McEstimate<f64>,
  /// Spot derivative from the orthogonal Malliavin weight.
  pub spot_delta: McEstimate<f64>,
  /// Second spot derivative from the conditional-lognormal Malliavin weight.
  pub spot_gamma: McEstimate<f64>,
  /// Covariance of antithetic-pair observations in price, delta, gamma order.
  pub sample_covariance:
    [[f64; HESTON_MALLIAVIN_SPOT_OBSERVABLES]; HESTON_MALLIAVIN_SPOT_OBSERVABLES],
  /// Covariance of the three reported Monte Carlo means.
  pub estimator_covariance:
    [[f64; HESTON_MALLIAVIN_SPOT_OBSERVABLES]; HESTON_MALLIAVIN_SPOT_OBSERVABLES],
  /// Number of raw simulated paths.
  pub paths: usize,
  /// Number of independent antithetic pairs used for standard errors.
  pub independent_samples: usize,
  /// Confirms that only unbumped antithetic paths were simulated.
  pub provenance: HestonMalliavinSpotProvenance,
}

pub(super) fn estimate_spot_greeks<P: TerminalPayoff + ?Sized, S: SeedExt>(
  model: HestonModel,
  config: &HestonMalliavinConfig<S>,
  payoff: &P,
) -> Result<HestonMalliavinSpotEstimate, QuantError> {
  let pairs = config.paths / 2;
  let mut normal = SimdNormal::<f64>::new(0.0, 1.0).seeded(&config.seed.clone());
  let mut statistics = OnlineCovariance::<HESTON_MALLIAVIN_SPOT_OBSERVABLES>::default();
  let mut variance_normals = vec![0.0; config.steps];
  let mut orthogonal_normals = vec![0.0; config.steps];
  let discount = (-model.risk_free_rate * model.tau).exp();

  for _ in 0..pairs {
    for draw in &mut variance_normals {
      *draw = normal.sample();
    }
    for draw in &mut orthogonal_normals {
      *draw = normal.sample();
    }
    let positive = spot_observation(
      model,
      config,
      payoff,
      1.0,
      discount,
      &variance_normals,
      &orthogonal_normals,
    )?;
    let negative = spot_observation(
      model,
      config,
      payoff,
      -1.0,
      discount,
      &variance_normals,
      &orthogonal_normals,
    )?;
    statistics.push(std::array::from_fn(|index| {
      0.5 * (positive[index] + negative[index])
    }));
  }
  let summary = statistics.finish()?;
  Ok(HestonMalliavinSpotEstimate {
    price: summary.estimate(0),
    spot_delta: summary.estimate(1),
    spot_gamma: summary.estimate(2),
    sample_covariance: summary.sample_covariance,
    estimator_covariance: summary.estimator_covariance,
    paths: config.paths,
    independent_samples: summary.independent_samples,
    provenance: HestonMalliavinSpotProvenance::BaseAntitheticPaths,
  })
}

fn spot_observation<P: TerminalPayoff + ?Sized, S: SeedExt>(
  model: HestonModel,
  config: &HestonMalliavinConfig<S>,
  payoff: &P,
  sign: f64,
  discount: f64,
  variance_normals: &[f64],
  orthogonal_normals: &[f64],
) -> Result<[f64; HESTON_MALLIAVIN_SPOT_OBSERVABLES], QuantError> {
  let path = simulate_path(
    model,
    model.params.v0,
    sign,
    variance_normals,
    orthogonal_normals,
    config.minimum_integrated_variance,
    config.minimum_conditional_variance,
  )?;
  let payoff_value = payoff.value(path.terminal_spot);
  if !payoff_value.is_finite() {
    return Err(non_finite_payoff(payoff_value, path.terminal_spot));
  }
  let discounted_payoff = discount * payoff_value;
  Ok([
    discounted_payoff,
    discounted_payoff * path.spot_delta_weight,
    discounted_payoff * path.spot_gamma_weight,
  ])
}
