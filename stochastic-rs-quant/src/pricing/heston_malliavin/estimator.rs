//! Seeded Monte Carlo estimation under correlated Heston dynamics.

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::normal::SimdNormal;

use super::HestonMalliavinError;
use super::TerminalPayoff;
use super::simulation::simulate_path;
use super::simulation::validate_config;
use super::simulation::validate_model;
use super::statistics::OnlineCovariance;
use super::statistics::OnlineScalar;
use super::variance_vega::DEFAULT_MINIMUM_RELATIVE_BUMP;
use super::variance_vega::HestonInitialVarianceVegaDiagnostics;
use super::variance_vega::classify_initial_variance_vega;
use super::variance_vega::effective_initial_variance_bump;

pub(super) const OBSERVABLES: usize = 4;

/// Risk-neutral Heston parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HestonModel {
  /// Initial underlying price.
  pub s: f64,
  /// Initial instantaneous variance.
  pub initial_variance: f64,
  /// Variance mean-reversion speed.
  pub kappa: f64,
  /// Long-run variance.
  pub theta: f64,
  /// Volatility of variance.
  pub vol_of_vol: f64,
  /// Spot/variance Brownian correlation.
  pub rho: f64,
  /// Continuously compounded risk-free rate.
  pub risk_free_rate: f64,
  /// Continuous dividend yield.
  pub dividend_yield: f64,
  /// Time to maturity in years.
  pub tau: f64,
}

/// Monte Carlo controls for the Malliavin estimator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HestonMalliavinConfig {
  /// Total raw path count, which must be even for antithetic pairing.
  pub paths: usize,
  /// Number of full-truncation log-Euler time steps.
  pub steps: usize,
  /// Reproducible random seed.
  pub seed: u64,
  /// Symmetric bump of initial variance used by the CRN variance vega.
  pub initial_variance_bump: f64,
  /// Minimum effective bump as a fraction of initial variance.
  ///
  /// The effective bump is the maximum of this relative floor and
  /// `initial_variance_bump`. It is returned with every estimate.
  pub minimum_relative_initial_variance_bump: f64,
  /// Smallest accepted integrated variance for the Malliavin covariance.
  pub minimum_integrated_variance: f64,
  /// Smallest accepted conditional log-spot variance from the orthogonal driver.
  pub minimum_conditional_variance: f64,
  /// Smallest accepted value of `1 - rho^2`.
  pub minimum_orthogonal_variance_fraction: f64,
}

impl Default for HestonMalliavinConfig {
  fn default() -> Self {
    Self {
      paths: 100_000,
      steps: 64,
      seed: 0x4853_544e_4d43,
      initial_variance_bump: 1e-4,
      minimum_relative_initial_variance_bump: DEFAULT_MINIMUM_RELATIVE_BUMP,
      minimum_integrated_variance: 1e-14,
      minimum_conditional_variance: 1e-10,
      minimum_orthogonal_variance_fraction: 1e-4,
    }
  }
}

/// One Monte Carlo estimate and its sampling standard error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EstimateWithError {
  /// Monte Carlo mean.
  pub value: f64,
  /// Standard error of the mean, accounting for antithetic pairing.
  pub standard_error: f64,
}

/// Joint price and state-Greek estimate.
#[derive(Debug, Clone, PartialEq)]
pub struct HestonMalliavinEstimate {
  /// Discounted option value.
  pub price: EstimateWithError,
  /// Spot derivative estimated with an exact Malliavin integration-by-parts weight.
  pub spot_delta: EstimateWithError,
  /// Second spot derivative from the exact conditional-lognormal Malliavin weight.
  pub spot_gamma: EstimateWithError,
  /// Initial-variance derivative estimated by centered common random numbers.
  pub initial_variance_vega: EstimateWithError,
  /// Effective bump and paired half-bump reliability evidence for the CRN vega.
  pub initial_variance_vega_diagnostics: HestonInitialVarianceVegaDiagnostics,
  /// Covariance of antithetic-pair observations in price, delta, gamma, vega order.
  pub sample_covariance: [[f64; OBSERVABLES]; OBSERVABLES],
  /// Covariance of the four reported Monte Carlo means.
  pub estimator_covariance: [[f64; OBSERVABLES]; OBSERVABLES],
  /// Number of raw simulated paths.
  pub paths: usize,
  /// Number of independent antithetic pairs used for standard errors.
  pub independent_samples: usize,
  /// Seed used by the simulation.
  pub seed: u64,
}

/// Raw path contribution for diagnostics and later conditional regressions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HestonMalliavinPathContribution {
  /// Identifier shared by the two antithetic paths.
  pub antithetic_pair: usize,
  /// `1` for the original normal draw and `-1` for its antithetic path.
  pub antithetic_sign: i8,
  /// Terminal underlying price on the base path.
  pub terminal_spot: f64,
  /// Discrete integrated variance on the base path.
  pub integrated_variance: f64,
  /// Integral of `sqrt(V)` against the Brownian direction orthogonal to variance.
  pub orthogonal_stochastic_integral: f64,
  /// Malliavin integration-by-parts weight for the spot derivative.
  pub spot_delta_weight: f64,
  /// Malliavin integration-by-parts weight for the second spot derivative.
  pub spot_gamma_weight: f64,
  /// Discounted payoff contribution to price.
  pub discounted_payoff: f64,
  /// Discounted Malliavin contribution to spot delta.
  pub spot_delta_contribution: f64,
  /// Discounted Malliavin contribution to spot gamma.
  pub spot_gamma_contribution: f64,
  /// Discounted CRN contribution to initial-variance vega.
  pub initial_variance_vega_contribution: f64,
  /// Discounted CRN contribution using half the effective variance bump.
  pub comparison_initial_variance_vega_contribution: f64,
}

/// Seeded estimator using the Brownian direction orthogonal to Heston variance.
///
/// Writing the spot Brownian as `rho dW_v + sqrt(1-rho^2) dW_perp`, the
/// variance path is independent of `W_perp`. For `A = integral V dt`, the exact
/// integration-by-parts weight of the discretized terminal spot is
/// `integral sqrt(V) dW_perp / (S0 sqrt(1-rho^2) A)`. It remains exact for
/// correlated Heston and non-smooth terminal payoffs, but becomes ill-conditioned
/// as `abs(rho)` approaches one.
///
/// Full truncation permits non-Feller parameter sets. If a path supplies too
/// little integrated or orthogonal conditional variance to invert safely, the
/// complete estimate fails closed instead of clipping the Malliavin weight.
#[derive(Debug, Clone, Copy)]
pub struct HestonMalliavinEstimator {
  model: HestonModel,
  config: HestonMalliavinConfig,
}

impl HestonMalliavinEstimator {
  /// Creates an estimator after validating model and simulation inputs.
  pub fn new(
    model: HestonModel,
    config: HestonMalliavinConfig,
  ) -> Result<Self, HestonMalliavinError> {
    validate_model(model)?;
    validate_config(model, config)?;
    Ok(Self { model, config })
  }

  /// Estimates price, Malliavin spot delta and gamma, and CRN initial-variance vega.
  pub fn estimate<P: TerminalPayoff + ?Sized>(
    &self,
    payoff: &P,
  ) -> Result<HestonMalliavinEstimate, HestonMalliavinError> {
    self.run(payoff, false).map(|(estimate, _)| estimate)
  }

  /// Estimates price, Malliavin spot delta, and gamma from base paths only.
  ///
  /// This skips every initial-variance-bumped path used by [`Self::estimate`].
  pub fn estimate_spot_greeks<P: TerminalPayoff + ?Sized>(
    &self,
    payoff: &P,
  ) -> Result<super::HestonMalliavinSpotEstimate, HestonMalliavinError> {
    super::spot_estimator::estimate_spot_greeks(self.model, self.config, payoff)
  }

  /// Estimates all observables and rejects an unresolved or bump-sensitive vega.
  pub fn estimate_requiring_stable_initial_variance_vega<P: TerminalPayoff + ?Sized>(
    &self,
    payoff: &P,
  ) -> Result<HestonMalliavinEstimate, HestonMalliavinError> {
    let estimate = self.estimate(payoff)?;
    if estimate
      .initial_variance_vega_diagnostics
      .stability
      .is_stable()
    {
      Ok(estimate)
    } else {
      Err(HestonMalliavinError::UnstableInitialVarianceVega)
    }
  }

  /// Estimates the observables and returns every raw path contribution.
  pub fn estimate_with_contributions<P: TerminalPayoff + ?Sized>(
    &self,
    payoff: &P,
  ) -> Result<
    (
      HestonMalliavinEstimate,
      Vec<HestonMalliavinPathContribution>,
    ),
    HestonMalliavinError,
  > {
    self.run(payoff, true)
  }

  fn run<P: TerminalPayoff + ?Sized>(
    &self,
    payoff: &P,
    retain_contributions: bool,
  ) -> Result<
    (
      HestonMalliavinEstimate,
      Vec<HestonMalliavinPathContribution>,
    ),
    HestonMalliavinError,
  > {
    let pairs = self.config.paths / 2;
    let normal = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(self.config.seed));
    let mut statistics = OnlineCovariance::<OBSERVABLES>::default();
    let mut comparison_statistics = OnlineScalar::default();
    let mut bump_difference_statistics = OnlineScalar::default();
    let mut contributions = if retain_contributions {
      Vec::with_capacity(self.config.paths)
    } else {
      Vec::new()
    };
    let mut variance_normals = vec![0.0; self.config.steps];
    let mut orthogonal_normals = vec![0.0; self.config.steps];

    for pair in 0..pairs {
      for draw in &mut variance_normals {
        *draw = normal.sample_fast();
      }
      for draw in &mut orthogonal_normals {
        *draw = normal.sample_fast();
      }

      let positive =
        self.path_contribution(payoff, pair, 1, &variance_normals, &orthogonal_normals)?;
      let negative =
        self.path_contribution(payoff, pair, -1, &variance_normals, &orthogonal_normals)?;
      statistics.push([
        0.5 * (positive.discounted_payoff + negative.discounted_payoff),
        0.5 * (positive.spot_delta_contribution + negative.spot_delta_contribution),
        0.5 * (positive.spot_gamma_contribution + negative.spot_gamma_contribution),
        0.5
          * (positive.initial_variance_vega_contribution
            + negative.initial_variance_vega_contribution),
      ]);
      let comparison = 0.5
        * (positive.comparison_initial_variance_vega_contribution
          + negative.comparison_initial_variance_vega_contribution);
      let primary = 0.5
        * (positive.initial_variance_vega_contribution
          + negative.initial_variance_vega_contribution);
      comparison_statistics.push(comparison)?;
      bump_difference_statistics.push(primary - comparison)?;

      if retain_contributions {
        contributions.push(positive);
        contributions.push(negative);
      }
    }

    let summary = statistics.finish()?;
    let initial_variance_vega = summary.estimate(3);
    let initial_variance_vega_diagnostics = classify_initial_variance_vega(
      self.config.initial_variance_bump,
      self.config.minimum_relative_initial_variance_bump,
      effective_initial_variance_bump(
        self.model.initial_variance,
        self.config.initial_variance_bump,
        self.config.minimum_relative_initial_variance_bump,
      )?,
      initial_variance_vega,
      comparison_statistics.finish()?,
      bump_difference_statistics.finish()?,
    );
    Ok((
      HestonMalliavinEstimate {
        price: summary.estimate(0),
        spot_delta: summary.estimate(1),
        spot_gamma: summary.estimate(2),
        initial_variance_vega,
        initial_variance_vega_diagnostics,
        sample_covariance: summary.sample_covariance,
        estimator_covariance: summary.estimator_covariance,
        paths: self.config.paths,
        independent_samples: summary.independent_samples,
        seed: self.config.seed,
      },
      contributions,
    ))
  }

  fn path_contribution<P: TerminalPayoff + ?Sized>(
    &self,
    payoff: &P,
    pair: usize,
    sign: i8,
    variance_normals: &[f64],
    orthogonal_normals: &[f64],
  ) -> Result<HestonMalliavinPathContribution, HestonMalliavinError> {
    let sign_value = f64::from(sign);
    let effective_bump = effective_initial_variance_bump(
      self.model.initial_variance,
      self.config.initial_variance_bump,
      self.config.minimum_relative_initial_variance_bump,
    )?;
    let comparison_bump = 0.5 * effective_bump;
    let base = simulate_path(
      self.model,
      self.model.initial_variance,
      sign_value,
      variance_normals,
      orthogonal_normals,
      self.config.minimum_integrated_variance,
      self.config.minimum_conditional_variance,
    )?;
    let up = simulate_path(
      self.model,
      self.model.initial_variance + effective_bump,
      sign_value,
      variance_normals,
      orthogonal_normals,
      self.config.minimum_integrated_variance,
      self.config.minimum_conditional_variance,
    )?;
    let down = simulate_path(
      self.model,
      self.model.initial_variance - effective_bump,
      sign_value,
      variance_normals,
      orthogonal_normals,
      self.config.minimum_integrated_variance,
      self.config.minimum_conditional_variance,
    )?;
    let comparison_up = simulate_path(
      self.model,
      self.model.initial_variance + comparison_bump,
      sign_value,
      variance_normals,
      orthogonal_normals,
      self.config.minimum_integrated_variance,
      self.config.minimum_conditional_variance,
    )?;
    let comparison_down = simulate_path(
      self.model,
      self.model.initial_variance - comparison_bump,
      sign_value,
      variance_normals,
      orthogonal_normals,
      self.config.minimum_integrated_variance,
      self.config.minimum_conditional_variance,
    )?;
    let discount = (-self.model.risk_free_rate * self.model.tau).exp();
    let base_payoff = finite_payoff(payoff, base.terminal_spot)?;
    let up_payoff = finite_payoff(payoff, up.terminal_spot)?;
    let down_payoff = finite_payoff(payoff, down.terminal_spot)?;
    let comparison_up_payoff = finite_payoff(payoff, comparison_up.terminal_spot)?;
    let comparison_down_payoff = finite_payoff(payoff, comparison_down.terminal_spot)?;
    let discounted_payoff = discount * base_payoff;
    let initial_variance_vega_contribution =
      discount * (up_payoff - down_payoff) / (2.0 * effective_bump);
    let comparison_initial_variance_vega_contribution =
      discount * (comparison_up_payoff - comparison_down_payoff) / (2.0 * comparison_bump);

    Ok(HestonMalliavinPathContribution {
      antithetic_pair: pair,
      antithetic_sign: sign,
      terminal_spot: base.terminal_spot,
      integrated_variance: base.integrated_variance,
      orthogonal_stochastic_integral: base.orthogonal_stochastic_integral,
      spot_delta_weight: base.spot_delta_weight,
      spot_gamma_weight: base.spot_gamma_weight,
      discounted_payoff,
      spot_delta_contribution: discounted_payoff * base.spot_delta_weight,
      spot_gamma_contribution: discounted_payoff * base.spot_gamma_weight,
      initial_variance_vega_contribution,
      comparison_initial_variance_vega_contribution,
    })
  }
}

fn finite_payoff<P: TerminalPayoff + ?Sized>(
  payoff: &P,
  terminal_spot: f64,
) -> Result<f64, HestonMalliavinError> {
  let value = payoff.value(terminal_spot);
  if value.is_finite() {
    Ok(value)
  } else {
    Err(HestonMalliavinError::NonFinitePayoff)
  }
}
