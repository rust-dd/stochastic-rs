//! # Swaption calibration of tree short-rate models
//!
//! Black–Karasinski and G2++ have no closed-form European swaption price, so
//! both calibrators reprice the quote grid on the crate's trinomial trees:
//! every quote becomes a one-exercise [`BermudanSwaption`] whose exercise
//! level and fixed-leg coupon levels are snapped to the grid, priced on a
//! [`BlackKarasinskiTree`] or [`G2ppTree`] rebuilt for each parameter trial.
//! The market side is the Black-76 at-the-money-forward price the Hull–White
//! calibrator uses, `A · black_forward_caplet(F, F, T, σ_B)`, and the
//! optimiser is the same Nelder–Mead on squared price residuals.
//!
//! The tree resolution (`steps_per_year`) trades accuracy for the cost of a
//! tree rebuild per quote and iteration — the G2++ lattice has
//! `(2L + 1)²` nodes on level `L`, so keep the grid modest there.
//!
//! References: Brigo, D. & Mercurio, F. (2006), *Interest Rate Models — Theory
//! and Practice*, 2nd ed., Springer, §3.5 (Black–Karasinski), §4.2 (G2++),
//! Ch. 13 (lattices); Hull, J. & White, A. (1994), *Numerical Procedures for
//! Implementing Term Structure Models I*, Journal of Derivatives 2(1), 7–16.

use std::convert::Infallible;

use basin::CostFunction;

use super::hw_swaption::CurveSnapshot;
use super::hw_swaption::MarketSwaption;
use super::hw_swaption::SwaptionQuote;
use super::hw_swaption::market_swaption;
use super::hw_swaption::serialize_curve;
use super::hw_swaption::validate_swaption_quotes;
use super::hw_swaption::weighted_sse;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationOptions;
use crate::calibration::Regularization;
use crate::calibration::nelder_mead_fit;
use crate::calibration::periodic_map;
use crate::calibration::quote::check_finite_start;
use crate::curves::DiscountCurve;
use crate::instruments::option::bermudan::BermudanSwaption;
use crate::instruments::option::types::ExerciseSchedule;
use crate::instruments::option::types::TreeCouponSchedule;
use crate::lattice::short_rate::BlackKarasinskiTree;
use crate::lattice::short_rate::BlackKarasinskiTreeModel;
use crate::lattice::short_rate::G2ppTree;
use crate::lattice::short_rate::G2ppTreeModel;
use crate::traits::ToShortRateModel;

/// Tree horizon, step count and the one-exercise swaption snapped to it.
fn tree_swaption(
  quote: &SwaptionQuote,
  market: &MarketSwaption,
  notional: f64,
  steps_per_year: usize,
) -> (f64, usize, BermudanSwaption<f64>) {
  let horizon = quote.expiry + quote.tenor;
  let steps = ((horizon * steps_per_year as f64).round() as usize).max(1);
  let dt = horizon / steps as f64;
  let level = |t: f64| ((t / dt).round() as usize).min(steps);
  let exercise = ExerciseSchedule::new(vec![level(quote.expiry)]);
  let coupons = TreeCouponSchedule::new(
    market.coupon_times.iter().map(|&t| level(t)).collect(),
    vec![market.accrual; market.coupon_times.len()],
  );
  (
    horizon,
    steps,
    BermudanSwaption::new(
      quote.direction,
      market.fair_rate,
      notional,
      exercise,
      coupons,
    ),
  )
}

fn assert_tree_inputs(notional: f64, steps_per_year: usize) {
  assert!(
    notional.is_finite() && notional > 0.0,
    "notional must satisfy `notional.is_finite() && notional > 0.0`, got notional = {notional:?}"
  );
  assert!(
    steps_per_year > 0,
    "steps_per_year must satisfy `steps_per_year > 0`, got steps_per_year = {steps_per_year:?}"
  );
}

/// Black–Karasinski parameters; the two rates shape the tree the fit prices on and stay fixed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlackKarasinskiParams {
  /// Mean reversion of the log-rate.
  pub mean_reversion: f64,
  /// Volatility of the log-rate.
  pub sigma: f64,
  /// Time-0 short rate of the tree model; fixed during the fit.
  pub initial_rate: f64,
  /// Long-run level the log-rate reverts to, in rate units; fixed during the fit.
  pub long_run_rate: f64,
}

impl BlackKarasinskiParams {
  /// `InvalidInput { what: "start" }` unless every field is finite and both rates positive.
  fn check(&self) -> Result<(), QuantError> {
    let rates = [
      ("initial_rate", self.initial_rate),
      ("long_run_rate", self.long_run_rate),
    ];
    check_finite_start(&[
      ("mean_reversion", self.mean_reversion),
      ("sigma", self.sigma),
    ])?;
    check_finite_start(&rates)?;
    match rates.into_iter().find(|&(_, rate)| rate <= 0.0) {
      Some((name, rate)) => Err(QuantError::invalid(
        "start",
        format!("{name} must satisfy `{name} > 0.0`, got {name} = {rate:?}"),
      )),
      None => Ok(()),
    }
  }
}

impl ToShortRateModel for Calibrated<BlackKarasinskiParams> {
  type Model = BlackKarasinskiTreeModel<f64>;

  fn to_short_rate_model(&self) -> Self::Model {
    let p = &self.params;
    BlackKarasinskiTreeModel::new(p.initial_rate, p.mean_reversion, p.long_run_rate, p.sigma)
  }
}

/// Black–Karasinski `(a, σ)` of `d ln r = a (ln θ − ln r) dt + σ dW` against a swaption grid.
#[derive(Debug, Clone)]
pub struct BlackKarasinskiSwaptionCalibrator<'a> {
  pub quotes: &'a [SwaptionQuote],
  pub curve: &'a DiscountCurve<f64>,
  pub notional: f64,
  /// Tree levels per year.
  pub steps_per_year: usize,
  pub start: BlackKarasinskiParams,
  /// Nelder–Mead: `max_iterations` (default 400), `tolerance` its spread; pull `(a, σ)`, and
  /// `patience` is `Unsupported`.
  pub options: CalibrationOptions,
}

impl<'a> BlackKarasinskiSwaptionCalibrator<'a> {
  /// Panics unless `notional > 0`, `steps_per_year > 0`, and the start is finite, its rates > 0.
  pub fn new(
    quotes: &'a [SwaptionQuote],
    curve: &'a DiscountCurve<f64>,
    notional: f64,
    steps_per_year: usize,
    start: BlackKarasinskiParams,
  ) -> Self {
    assert_tree_inputs(notional, steps_per_year);
    start.check().unwrap_or_else(|e| panic!("{e}"));
    Self {
      quotes,
      curve,
      notional,
      steps_per_year,
      start,
      options: CalibrationOptions::default(),
    }
  }

  pub fn with_options(mut self, options: CalibrationOptions) -> Self {
    self.options = options;
    self
  }

  fn cost(
    &self,
    start: &BlackKarasinskiParams,
    regularization: Option<Regularization>,
  ) -> BlackKarasinskiCost {
    BlackKarasinskiCost {
      quotes: self.quotes.to_vec(),
      curve_points: serialize_curve(self.curve),
      notional: self.notional,
      initial_rate: start.initial_rate,
      long_run_rate: start.long_run_rate,
      steps_per_year: self.steps_per_year,
      regularization,
    }
  }
}

impl crate::traits::Calibrator for BlackKarasinskiSwaptionCalibrator<'_> {
  type Params = BlackKarasinskiParams;
  type Output = Calibrated<BlackKarasinskiParams>;
  type Error = QuantError;

  /// A start replaces `self.start` whole, the two rates included; `InvalidInput` unless it is
  /// finite with positive rates, and the `CalibrationOptions` errors.
  fn calibrate(&self, initial: Option<BlackKarasinskiParams>) -> Result<Self::Output, QuantError> {
    let start = initial.unwrap_or(self.start);
    start.check()?;
    validate_swaption_quotes(self.quotes, self.curve)?;
    let problem = self.cost(&start, self.options.regularization_for(2)?);
    let (a0, s0) = (start.mean_reversion, start.sigma);
    let simplex = vec![vec![a0, s0], vec![a0 * 1.5, s0], vec![a0, s0 * 1.5]];
    nelder_mead_fit(problem.clone(), simplex, &self.options, 400, |x| {
      let (mean_reversion, sigma) = (x[0].abs().max(1e-6), x[1].abs().max(1e-6));
      let (market, model) = problem.price_series(mean_reversion, sigma);
      let params = BlackKarasinskiParams {
        mean_reversion,
        sigma,
        ..start
      };
      (params, market, model)
    })
  }
}

#[derive(Clone)]
struct BlackKarasinskiCost {
  quotes: Vec<SwaptionQuote>,
  curve_points: CurveSnapshot,
  notional: f64,
  initial_rate: f64,
  long_run_rate: f64,
  steps_per_year: usize,
  regularization: Option<Regularization>,
}

impl BlackKarasinskiCost {
  fn price_series(&self, a: f64, sigma: f64) -> (Vec<f64>, Vec<f64>) {
    let curve = self.curve_points.rebuild();
    let model = BlackKarasinskiTreeModel::new(self.initial_rate, a, self.long_run_rate, sigma);
    let mut model_prices = Vec::with_capacity(self.quotes.len());
    let mut market_prices = Vec::with_capacity(self.quotes.len());
    for quote in &self.quotes {
      let market = market_swaption(quote, &curve, self.notional);
      let (horizon, steps, swaption) =
        tree_swaption(quote, &market, self.notional, self.steps_per_year);
      let tree = BlackKarasinskiTree::new(model.clone(), horizon, steps);
      model_prices.push(swaption.price_on_tree(&tree.tree, &tree.model));
      market_prices.push(market.market_price);
    }
    (market_prices, model_prices)
  }
}

impl CostFunction for BlackKarasinskiCost {
  type Param = Vec<f64>;
  type Output = f64;
  type Error = Infallible;

  fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
    let (a, sigma) = (x[0].abs().max(1e-6), x[1].abs().max(1e-6));
    let (market, model) = self.price_series(a, sigma);
    let penalty = self
      .regularization
      .as_ref()
      .map_or(0.0, |reg| reg.penalty(&[a, sigma]));
    Ok(weighted_sse(&self.quotes, &market, &model) + penalty)
  }
}

/// G2++ `(a, b, σ, η, ρ)` and the shift `phi` the fit prices on and keeps fixed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct G2ppParams {
  pub mean_reversion_x: f64,
  pub mean_reversion_y: f64,
  pub sigma_x: f64,
  pub sigma_y: f64,
  pub rho: f64,
  /// Constant shift φ with both factors started at zero; fixed during the fit.
  pub phi: f64,
}

impl G2ppParams {
  fn fields(&self) -> [(&'static str, f64); 6] {
    [
      ("mean_reversion_x", self.mean_reversion_x),
      ("mean_reversion_y", self.mean_reversion_y),
      ("sigma_x", self.sigma_x),
      ("sigma_y", self.sigma_y),
      ("rho", self.rho),
      ("phi", self.phi),
    ]
  }
}

impl ToShortRateModel for Calibrated<G2ppParams> {
  type Model = G2ppTreeModel<f64>;

  fn to_short_rate_model(&self) -> Self::Model {
    let p = &self.params;
    G2ppTreeModel::new(
      0.0,
      0.0,
      p.phi,
      p.mean_reversion_x,
      p.mean_reversion_y,
      p.sigma_x,
      p.sigma_y,
      p.rho,
    )
  }
}

/// G2++ calibrator: `(a, b, σ, η, ρ)` of the two-factor Gaussian model against
/// a swaption grid on the two-dimensional trinomial tree.
#[derive(Debug, Clone)]
pub struct G2ppSwaptionCalibrator<'a> {
  pub quotes: &'a [SwaptionQuote],
  pub curve: &'a DiscountCurve<f64>,
  pub notional: f64,
  /// Tree levels per year.
  pub steps_per_year: usize,
  pub start: G2ppParams,
  /// Nelder–Mead: `max_iterations` (default 400), `tolerance` its spread; pull `(a, b, σ, η, ρ)`,
  /// and `patience` is `Unsupported`.
  pub options: CalibrationOptions,
}

impl<'a> G2ppSwaptionCalibrator<'a> {
  /// Panics unless `notional > 0`, `steps_per_year > 0` and the start is finite.
  pub fn new(
    quotes: &'a [SwaptionQuote],
    curve: &'a DiscountCurve<f64>,
    notional: f64,
    steps_per_year: usize,
    start: G2ppParams,
  ) -> Self {
    assert_tree_inputs(notional, steps_per_year);
    check_finite_start(&start.fields()).unwrap_or_else(|e| panic!("{e}"));
    Self {
      quotes,
      curve,
      notional,
      steps_per_year,
      start,
      options: CalibrationOptions::default(),
    }
  }

  pub fn with_options(mut self, options: CalibrationOptions) -> Self {
    self.options = options;
    self
  }

  fn cost(&self, phi: f64, regularization: Option<Regularization>) -> G2ppCost {
    G2ppCost {
      quotes: self.quotes.to_vec(),
      curve_points: serialize_curve(self.curve),
      notional: self.notional,
      phi,
      steps_per_year: self.steps_per_year,
      regularization,
    }
  }
}

impl crate::traits::Calibrator for G2ppSwaptionCalibrator<'_> {
  type Params = G2ppParams;
  type Output = Calibrated<G2ppParams>;
  type Error = QuantError;

  /// A start replaces `self.start` whole, `phi` included; `InvalidInput` unless it is finite,
  /// and the `CalibrationOptions` errors.
  fn calibrate(&self, initial: Option<G2ppParams>) -> Result<Self::Output, QuantError> {
    let start = initial.unwrap_or(self.start);
    check_finite_start(&start.fields())?;
    validate_swaption_quotes(self.quotes, self.curve)?;
    let problem = self.cost(start.phi, self.options.regularization_for(5)?);
    let x0 = [
      start.mean_reversion_x,
      start.mean_reversion_y,
      start.sigma_x,
      start.sigma_y,
      start.rho,
    ];
    let mut simplex = vec![x0.to_vec()];
    for i in 0..5 {
      let mut v = x0.to_vec();
      v[i] = if i == 4 {
        x0[4] - 0.25_f64.copysign(x0[4])
      } else {
        x0[i] * 1.5
      };
      simplex.push(v);
    }
    nelder_mead_fit(problem.clone(), simplex, &self.options, 400, |x| {
      let params = problem.params(x);
      let (market, model) = problem.price_series(&params);
      (params, market, model)
    })
  }
}

#[derive(Clone)]
struct G2ppCost {
  quotes: Vec<SwaptionQuote>,
  curve_points: CurveSnapshot,
  notional: f64,
  phi: f64,
  steps_per_year: usize,
  regularization: Option<Regularization>,
}

impl G2ppCost {
  /// Positive reversions and volatilities; a correlation past ±0.999 is reflected back, since a
  /// clamp would leave the simplex a flat face to collapse on.
  fn params(&self, x: &[f64]) -> G2ppParams {
    G2ppParams {
      mean_reversion_x: x[0].abs().max(1e-6),
      mean_reversion_y: x[1].abs().max(1e-6),
      sigma_x: x[2].abs().max(1e-6),
      sigma_y: x[3].abs().max(1e-6),
      rho: periodic_map(x[4], -0.999, 0.999),
      phi: self.phi,
    }
  }

  fn price_series(&self, p: &G2ppParams) -> (Vec<f64>, Vec<f64>) {
    let curve = self.curve_points.rebuild();
    let model = G2ppTreeModel::new(
      0.0,
      0.0,
      p.phi,
      p.mean_reversion_x,
      p.mean_reversion_y,
      p.sigma_x,
      p.sigma_y,
      p.rho,
    );
    let mut model_prices = Vec::with_capacity(self.quotes.len());
    let mut market_prices = Vec::with_capacity(self.quotes.len());
    for quote in &self.quotes {
      let market = market_swaption(quote, &curve, self.notional);
      let (horizon, steps, swaption) =
        tree_swaption(quote, &market, self.notional, self.steps_per_year);
      let tree = G2ppTree::new(model.clone(), horizon, steps);
      model_prices.push(swaption.price_on_g2pp(&tree));
      market_prices.push(market.market_price);
    }
    (market_prices, model_prices)
  }
}

impl CostFunction for G2ppCost {
  type Param = Vec<f64>;
  type Output = f64;
  type Error = Infallible;

  fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
    let p = self.params(x);
    let (market, model) = self.price_series(&p);
    let penalty = self.regularization.as_ref().map_or(0.0, |reg| {
      reg.penalty(&[
        p.mean_reversion_x,
        p.mean_reversion_y,
        p.sigma_x,
        p.sigma_y,
        p.rho,
      ])
    });
    Ok(weighted_sse(&self.quotes, &market, &model) + penalty)
  }
}

#[cfg(test)]
mod tests;
