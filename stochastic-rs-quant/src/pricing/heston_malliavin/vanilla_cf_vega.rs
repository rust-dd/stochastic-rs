//! Deterministic characteristic-function variance Greek for vanilla portfolios.

use super::HestonModel;
use super::VanillaLeg;
use super::VanillaPortfolio;
use super::invalid_input;
use super::simulation::assert_model;
use super::variance_vega::checked_initial_variance_bump;
use crate::QuantError;
use crate::pricing::heston::HestonParams;
use crate::pricing::heston::HestonPricer;
use crate::traits::ModelPricer;

/// Bump controls for deterministic Heston characteristic-function vega.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HestonVanillaCfVegaConfig {
  /// Requested absolute initial-variance bump.
  pub initial_variance_bump: f64,
  /// Minimum bump as a fraction of initial variance.
  pub minimum_relative_initial_variance_bump: f64,
  /// Maximum relative main-versus-half-bump difference classified as stable.
  pub maximum_relative_bump_difference: f64,
}

/// Primary method used for the reported initial-variance derivative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HestonVanillaCfVegaMethod {
  /// Analytic differentiation of the Heston characteristic-function integral.
  AnalyticCharacteristicFunction,
}

impl Default for HestonVanillaCfVegaConfig {
  fn default() -> Self {
    Self {
      initial_variance_bump: 1e-5,
      minimum_relative_initial_variance_bump: 1e-4,
      maximum_relative_bump_difference: 1e-4,
    }
  }
}

/// Deterministic Heston CF initial-variance derivative and bump provenance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HestonVanillaCfVegaEstimate {
  /// Characteristic-function value of the portfolio at the unbumped state.
  pub base_price: f64,
  /// Analytic characteristic-function derivative with respect to initial variance.
  pub value: f64,
  /// Method used to compute `value`.
  pub method: HestonVanillaCfVegaMethod,
  /// Absolute bump requested by the caller.
  pub requested_bump: f64,
  /// Bump actually used after applying the relative floor.
  pub effective_bump: f64,
  /// Centered finite-difference diagnostic evaluated at `effective_bump`.
  pub finite_difference_value: f64,
  /// Half-sized bump used for the deterministic stability check.
  pub comparison_bump: f64,
  /// Centered derivative evaluated at `comparison_bump`.
  pub comparison_value: f64,
  /// Relative difference between the two finite-difference diagnostics.
  pub relative_bump_difference: f64,
  /// Whether the non-primary finite-difference comparison meets its tolerance.
  pub bump_stable: bool,
}

/// Analytic Heston CF `∂price/∂v0` of a linear call/put portfolio, the centred differences beside
/// it diagnostics only; panics on a `model` outside [`HestonModel`]'s domain.
pub fn heston_vanilla_portfolio_cf_initial_variance_vega(
  model: HestonModel,
  portfolio: &VanillaPortfolio,
  config: HestonVanillaCfVegaConfig,
) -> Result<HestonVanillaCfVegaEstimate, QuantError> {
  assert_model(model);
  let tolerance = config.maximum_relative_bump_difference;
  if !tolerance.is_finite() || tolerance < 0.0 {
    return Err(invalid_input(format!(
      "maximum_relative_bump_difference must be finite and non-negative, got {tolerance}"
    )));
  }
  let effective_bump = checked_initial_variance_bump(
    model.params.v0,
    config.initial_variance_bump,
    config.minimum_relative_initial_variance_bump,
  )?;
  let comparison_bump = 0.5 * effective_bump;
  let base_price = portfolio_price(model, portfolio, model.params.v0)?;
  let value = portfolio_initial_variance_vega(model, portfolio)?;
  let finite_difference_value = centered_derivative(model, portfolio, effective_bump)?;
  let comparison_value = centered_derivative(model, portfolio, comparison_bump)?;
  let scale = finite_difference_value
    .abs()
    .max(comparison_value.abs())
    .max(f64::MIN_POSITIVE);
  let relative_bump_difference = (finite_difference_value - comparison_value).abs() / scale;
  Ok(HestonVanillaCfVegaEstimate {
    base_price,
    value,
    method: HestonVanillaCfVegaMethod::AnalyticCharacteristicFunction,
    requested_bump: config.initial_variance_bump,
    effective_bump,
    finite_difference_value,
    comparison_bump,
    comparison_value,
    relative_bump_difference,
    bump_stable: relative_bump_difference <= config.maximum_relative_bump_difference,
  })
}

fn portfolio_initial_variance_vega(
  model: HestonModel,
  portfolio: &VanillaPortfolio,
) -> Result<f64, QuantError> {
  let mut value = 0.0;
  for (index, leg) in portfolio.legs().iter().enumerate() {
    check_leg(index, leg)?;
    let (pricer, s, k, r, q, tau) = pricer(model, leg.strike, model.params.v0);
    value += leg.quantity * pricer.initial_variance_vega(s, k, r, q, tau);
  }
  if value.is_finite() {
    Ok(value)
  } else {
    Err(QuantError::Numerical {
      what: "the characteristic-function vega",
      reason: format!("the portfolio vega is {value}"),
    })
  }
}

fn centered_derivative(
  model: HestonModel,
  portfolio: &VanillaPortfolio,
  bump: f64,
) -> Result<f64, QuantError> {
  let up = portfolio_price(model, portfolio, model.params.v0 + bump)?;
  let down = portfolio_price(model, portfolio, model.params.v0 - bump)?;
  let derivative = (up - down) / (2.0 * bump);
  if derivative.is_finite() {
    Ok(derivative)
  } else {
    Err(QuantError::Numerical {
      what: "the finite-difference vega",
      reason: format!("the centered derivative at bump {bump} is {derivative}"),
    })
  }
}

fn portfolio_price(
  model: HestonModel,
  portfolio: &VanillaPortfolio,
  initial_variance: f64,
) -> Result<f64, QuantError> {
  let mut price = 0.0;
  for (index, leg) in portfolio.legs().iter().enumerate() {
    check_leg(index, leg)?;
    let (pricer, s, k, r, q, tau) = pricer(model, leg.strike, initial_variance);
    price += leg.quantity * pricer.price_option(s, k, r, q, tau, leg.kind);
  }
  if price.is_finite() {
    Ok(price)
  } else {
    Err(QuantError::Numerical {
      what: "the portfolio price",
      reason: format!("it is {price} at initial variance {initial_variance}"),
    })
  }
}

fn check_leg(index: usize, leg: &VanillaLeg) -> Result<(), QuantError> {
  if !leg.strike.is_finite() || leg.strike <= 0.0 || !leg.quantity.is_finite() {
    return Err(invalid_input(format!(
      "vanilla leg {index} needs a finite positive strike and a finite quantity, got strike {}, \
       quantity {}",
      leg.strike, leg.quantity
    )));
  }
  Ok(())
}

/// The Heston model at `initial_variance`, paired with the
/// `(s, k, r, q, tau)` query this portfolio leg is priced at.
fn pricer(
  model: HestonModel,
  strike: f64,
  initial_variance: f64,
) -> (HestonPricer, f64, f64, f64, f64, f64) {
  (
    HestonPricer::new(HestonParams {
      v0: initial_variance,
      ..model.params
    }),
    model.s,
    strike,
    model.risk_free_rate,
    model.dividend_yield,
    model.tau,
  )
}
