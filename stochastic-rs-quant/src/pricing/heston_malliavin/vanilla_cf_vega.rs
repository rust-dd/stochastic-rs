//! Deterministic characteristic-function variance Greek for vanilla portfolios.

use super::HestonMalliavinError;
use super::HestonModel;
use super::VanillaPortfolio;
use super::simulation::validate_model;
use super::variance_vega::effective_initial_variance_bump;
use crate::OptionType;
use crate::pricing::heston::HestonPricer;
use crate::traits::PricerExt;

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

/// Computes a deterministic Heston CF `v0` derivative for a vanilla portfolio.
///
/// This helper is limited to linear portfolios of European calls and puts. Its
/// primary value differentiates the characteristic-function integral
/// analytically. The retained centered finite differences are compatibility
/// diagnostics and do not determine whether the analytic value is available.
pub fn heston_vanilla_portfolio_cf_initial_variance_vega(
  model: HestonModel,
  portfolio: &VanillaPortfolio,
  config: HestonVanillaCfVegaConfig,
) -> Result<HestonVanillaCfVegaEstimate, HestonMalliavinError> {
  validate_model(model)?;
  if model.vol_of_vol <= 0.0 {
    return Err(HestonMalliavinError::InvalidInput(
      "Heston CF vega requires positive vol_of_vol",
    ));
  }
  if !config.maximum_relative_bump_difference.is_finite()
    || config.maximum_relative_bump_difference < 0.0
  {
    return Err(HestonMalliavinError::InvalidInput(
      "maximum_relative_bump_difference must be finite and non-negative",
    ));
  }
  let effective_bump = effective_initial_variance_bump(
    model.initial_variance,
    config.initial_variance_bump,
    config.minimum_relative_initial_variance_bump,
  )?;
  let comparison_bump = 0.5 * effective_bump;
  let base_price = portfolio_price(model, portfolio, model.initial_variance)?;
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
) -> Result<f64, HestonMalliavinError> {
  let mut value = 0.0;
  for leg in portfolio.legs() {
    if !leg.strike.is_finite() || leg.strike <= 0.0 || !leg.quantity.is_finite() {
      return Err(HestonMalliavinError::InvalidInput(
        "vanilla legs require positive strikes and finite quantities",
      ));
    }
    let pricer = pricer(model, leg.strike, model.initial_variance);
    let (call, put) = pricer.calculate_call_put_initial_variance_vega();
    let leg_value = match leg.kind {
      OptionType::Call => call,
      OptionType::Put => put,
    };
    value += leg.quantity * leg_value;
  }
  if value.is_finite() {
    Ok(value)
  } else {
    Err(HestonMalliavinError::NonFiniteSimulation)
  }
}

fn centered_derivative(
  model: HestonModel,
  portfolio: &VanillaPortfolio,
  bump: f64,
) -> Result<f64, HestonMalliavinError> {
  let up = portfolio_price(model, portfolio, model.initial_variance + bump)?;
  let down = portfolio_price(model, portfolio, model.initial_variance - bump)?;
  let derivative = (up - down) / (2.0 * bump);
  if derivative.is_finite() {
    Ok(derivative)
  } else {
    Err(HestonMalliavinError::NonFiniteSimulation)
  }
}

fn portfolio_price(
  model: HestonModel,
  portfolio: &VanillaPortfolio,
  initial_variance: f64,
) -> Result<f64, HestonMalliavinError> {
  let mut price = 0.0;
  for leg in portfolio.legs() {
    if !leg.strike.is_finite() || leg.strike <= 0.0 || !leg.quantity.is_finite() {
      return Err(HestonMalliavinError::InvalidInput(
        "vanilla legs require positive strikes and finite quantities",
      ));
    }
    let pricer = pricer(model, leg.strike, initial_variance);
    let (call, put) = pricer.calculate_call_put();
    let leg_price = match leg.kind {
      OptionType::Call => call,
      OptionType::Put => put,
    };
    price += leg.quantity * leg_price;
  }
  if price.is_finite() {
    Ok(price)
  } else {
    Err(HestonMalliavinError::NonFiniteSimulation)
  }
}

fn pricer(model: HestonModel, strike: f64, initial_variance: f64) -> HestonPricer {
  HestonPricer::builder(
    model.spot,
    initial_variance,
    strike,
    model.risk_free_rate,
    model.rho,
    model.kappa,
    model.theta,
    model.vol_of_vol,
  )
  .q(model.dividend_yield)
  .tau(model.maturity)
  .build()
}
