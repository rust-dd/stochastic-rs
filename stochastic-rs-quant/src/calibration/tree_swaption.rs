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

use argmin::core::CostFunction;
use argmin::core::Executor;
use argmin::core::State;
use argmin::solver::neldermead::NelderMead;

use super::hw_swaption::CurveSnapshot;
use super::hw_swaption::SwaptionQuote;
use super::hw_swaption::serialize_curve;
use crate::calibration::Regularization;
use crate::curves::DiscountCurve;
use crate::instruments::option::bermudan::BermudanSwaption;
use crate::instruments::option::caplet::black_forward_caplet;
use crate::instruments::option::types::ExerciseSchedule;
use crate::instruments::option::types::TreeCouponSchedule;
use crate::lattice::short_rate::BlackKarasinskiTree;
use crate::lattice::short_rate::BlackKarasinskiTreeModel;
use crate::lattice::short_rate::G2ppTree;
use crate::lattice::short_rate::G2ppTreeModel;

/// Market side of one quote: the forward swap and its Black-76 price.
#[derive(Debug, Clone)]
struct MarketSwaption {
  fair_rate: f64,
  coupon_times: Vec<f64>,
  accrual: f64,
  market_price: f64,
}

fn market_swaption(
  quote: &SwaptionQuote,
  curve: &DiscountCurve<f64>,
  notional: f64,
) -> MarketSwaption {
  let payments_per_year = (1.0 / quote.fixed_accrual).round();
  let n_payments = ((quote.tenor * payments_per_year).round() as usize).max(1);
  let accrual = quote.tenor / n_payments as f64;
  let coupon_times: Vec<f64> = (1..=n_payments)
    .map(|k| quote.expiry + accrual * k as f64)
    .collect();
  let annuity: f64 = coupon_times
    .iter()
    .map(|&t| curve.discount_factor(t) * accrual)
    .sum::<f64>()
    * notional;
  let p_exp = curve.discount_factor(quote.expiry);
  let p_end = curve.discount_factor(*coupon_times.last().expect("non-empty"));
  let fair_rate = (p_exp - p_end) / (annuity / notional).max(1e-14);
  let market_price =
    annuity * black_forward_caplet(fair_rate, fair_rate, quote.expiry, quote.black_vol);
  MarketSwaption {
    fair_rate,
    coupon_times,
    accrual,
    market_price,
  }
}

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

fn rmse(model: &[f64], market: &[f64]) -> f64 {
  let n = model.len().max(1) as f64;
  (model
    .iter()
    .zip(market)
    .map(|(m, q)| (m - q).powi(2))
    .sum::<f64>()
    / n)
    .sqrt()
}

fn run_nelder_mead<C: CostFunction<Param = Vec<f64>, Output = f64> + Clone>(
  problem: C,
  simplex: Vec<Vec<f64>>,
  max_iters: u64,
  sd_tolerance: f64,
) -> (Vec<f64>, bool) {
  match NelderMead::new(simplex.clone()).with_sd_tolerance(sd_tolerance) {
    Ok(solver) => match Executor::new(problem, solver)
      .configure(|s| s.max_iters(max_iters))
      .run()
    {
      Ok(res) => (
        res
          .state
          .get_best_param()
          .cloned()
          .unwrap_or_else(|| simplex[0].clone()),
        true,
      ),
      Err(_) => (simplex[0].clone(), false),
    },
    Err(_) => (simplex[0].clone(), false),
  }
}

/// Calibrated Black–Karasinski parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct BlackKarasinskiParams {
  /// Mean reversion of the log-rate.
  pub mean_reversion: f64,
  /// Volatility of the log-rate.
  pub sigma: f64,
}

/// Black–Karasinski swaption calibration result.
#[derive(Debug, Clone)]
pub struct BlackKarasinskiCalibrationResult {
  pub mean_reversion: f64,
  pub sigma: f64,
  /// Root-mean-square price error on the quote grid.
  pub rmse: f64,
  /// True when Nelder–Mead reported convergence.
  pub converged: bool,
  pub model_prices: Vec<f64>,
  pub market_prices: Vec<f64>,
}

impl BlackKarasinskiCalibrationResult {
  /// Tree model at the calibrated `(a, σ)`; `long_run_rate` is the level
  /// the log-rate reverts to, in rate units.
  pub fn to_short_rate_model(
    &self,
    initial_rate: f64,
    long_run_rate: f64,
  ) -> BlackKarasinskiTreeModel<f64> {
    BlackKarasinskiTreeModel::new(initial_rate, self.mean_reversion, long_run_rate, self.sigma)
  }
}

impl crate::traits::ToShortRateModel for BlackKarasinskiCalibrationResult {
  type Model = BlackKarasinskiTreeModel<f64>;
  /// `theta` is the long-run short rate (rate units, not its log).
  fn to_short_rate_model(&self, initial_rate: f64, theta: f64) -> Self::Model {
    BlackKarasinskiCalibrationResult::to_short_rate_model(self, initial_rate, theta)
  }
}

impl crate::traits::CalibrationResult for BlackKarasinskiCalibrationResult {
  type Params = BlackKarasinskiParams;
  fn rmse(&self) -> f64 {
    self.rmse
  }
  fn converged(&self) -> bool {
    self.converged
  }
  fn params(&self) -> Self::Params {
    BlackKarasinskiParams {
      mean_reversion: self.mean_reversion,
      sigma: self.sigma,
    }
  }
}

/// Black–Karasinski calibrator: `(a, σ)` of `d ln r = a (ln θ − ln r) dt + σ dW`
/// against a swaption grid, with the initial and long-run rates given.
#[derive(Debug, Clone)]
pub struct BlackKarasinskiSwaptionCalibrator<'a> {
  pub quotes: &'a [SwaptionQuote],
  pub curve: &'a DiscountCurve<f64>,
  pub notional: f64,
  pub initial_rate: f64,
  pub long_run_rate: f64,
  /// Tree levels per year.
  pub steps_per_year: usize,
  pub initial_guess: Option<(f64, f64)>,
  pub max_iters: u64,
  pub sd_tolerance: f64,
  /// Optional Tikhonov pull of `(a, σ)` toward an anchor.
  pub regularization: Option<Regularization>,
}

impl<'a> BlackKarasinskiSwaptionCalibrator<'a> {
  pub fn new(
    quotes: &'a [SwaptionQuote],
    curve: &'a DiscountCurve<f64>,
    notional: f64,
    initial_rate: f64,
    long_run_rate: f64,
    steps_per_year: usize,
  ) -> Self {
    Self {
      quotes,
      curve,
      notional,
      initial_rate,
      long_run_rate,
      steps_per_year,
      initial_guess: None,
      max_iters: 400,
      sd_tolerance: 1e-10,
      regularization: None,
    }
  }

  /// Overrides the Nelder–Mead iteration cap.
  pub fn with_max_iters(mut self, max_iters: u64) -> Self {
    self.max_iters = max_iters;
    self
  }

  /// Adds a Tikhonov pull toward `regularization.anchor` in the order `(a, σ)`.
  pub fn with_regularization(mut self, regularization: Regularization) -> Self {
    assert_eq!(
      regularization.dimension(),
      2,
      "Black-Karasinski regularisation needs two anchors"
    );
    self.regularization = Some(regularization);
    self
  }

  fn cost(&self) -> BlackKarasinskiCost {
    BlackKarasinskiCost {
      quotes: self.quotes.to_vec(),
      curve_points: serialize_curve(self.curve),
      notional: self.notional,
      initial_rate: self.initial_rate,
      long_run_rate: self.long_run_rate,
      steps_per_year: self.steps_per_year,
      regularization: self.regularization.clone(),
    }
  }

  fn solve(&self) -> BlackKarasinskiCalibrationResult {
    let problem = self.cost();
    let (a0, s0) = self.initial_guess.unwrap_or((0.1, 0.2));
    let simplex = vec![vec![a0, s0], vec![a0 * 1.5, s0], vec![a0, s0 * 1.5]];
    let (best, converged) =
      run_nelder_mead(problem.clone(), simplex, self.max_iters, self.sd_tolerance);
    let (a, sigma) = (best[0].abs().max(1e-6), best[1].abs().max(1e-6));
    let (model_prices, market_prices) = problem.price_series(a, sigma);
    BlackKarasinskiCalibrationResult {
      mean_reversion: a,
      sigma,
      rmse: rmse(&model_prices, &market_prices),
      converged,
      model_prices,
      market_prices,
    }
  }
}

impl<'a> crate::traits::Calibrator for BlackKarasinskiSwaptionCalibrator<'a> {
  type InitialGuess = (f64, f64);
  type Params = BlackKarasinskiParams;
  type Output = BlackKarasinskiCalibrationResult;
  type Error = anyhow::Error;

  fn calibrate(&self, initial: Option<Self::InitialGuess>) -> Result<Self::Output, Self::Error> {
    let mut this = self.clone();
    if let Some(guess) = initial {
      this.initial_guess = Some(guess);
    }
    Ok(this.solve())
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
      let w = quote.weight.unwrap_or(1.0);
      model_prices.push(w * swaption.price_on_tree(&tree.tree, &tree.model));
      market_prices.push(w * market.market_price);
    }
    (model_prices, market_prices)
  }
}

impl CostFunction for BlackKarasinskiCost {
  type Param = Vec<f64>;
  type Output = f64;
  fn cost(&self, x: &Self::Param) -> Result<f64, argmin::core::Error> {
    let (a, sigma) = (x[0].abs().max(1e-6), x[1].abs().max(1e-6));
    let (model, market) = self.price_series(a, sigma);
    let penalty = self
      .regularization
      .as_ref()
      .map_or(0.0, |reg| reg.penalty(&[a, sigma]));
    Ok(
      model
        .iter()
        .zip(&market)
        .map(|(m, q)| (m - q).powi(2))
        .sum::<f64>()
        + penalty,
    )
  }
}

/// Calibrated G2++ parameters `(a, b, σ, η, ρ)`.
#[derive(Debug, Clone, PartialEq)]
pub struct G2ppParams {
  pub mean_reversion_x: f64,
  pub mean_reversion_y: f64,
  pub sigma_x: f64,
  pub sigma_y: f64,
  pub rho: f64,
}

/// G2++ swaption calibration result.
#[derive(Debug, Clone)]
pub struct G2ppCalibrationResult {
  pub params: G2ppParams,
  /// Root-mean-square price error on the quote grid.
  pub rmse: f64,
  /// True when Nelder–Mead reported convergence.
  pub converged: bool,
  pub model_prices: Vec<f64>,
  pub market_prices: Vec<f64>,
}

impl G2ppCalibrationResult {
  /// Tree model at the calibrated parameters with both factors starting at
  /// zero and the constant shift `φ = initial_rate`, so `r = x + y + φ`.
  pub fn to_short_rate_model(&self, initial_rate: f64) -> G2ppTreeModel<f64> {
    let p = &self.params;
    G2ppTreeModel::new(
      0.0,
      0.0,
      initial_rate,
      p.mean_reversion_x,
      p.mean_reversion_y,
      p.sigma_x,
      p.sigma_y,
      p.rho,
    )
  }
}

impl crate::traits::ToShortRateModel for G2ppCalibrationResult {
  type Model = G2ppTreeModel<f64>;
  /// The constant shift is `initial_rate`; `theta` has no role in G2++ and
  /// is ignored.
  fn to_short_rate_model(&self, initial_rate: f64, _theta: f64) -> Self::Model {
    G2ppCalibrationResult::to_short_rate_model(self, initial_rate)
  }
}

impl crate::traits::CalibrationResult for G2ppCalibrationResult {
  type Params = G2ppParams;
  fn rmse(&self) -> f64 {
    self.rmse
  }
  fn converged(&self) -> bool {
    self.converged
  }
  fn params(&self) -> Self::Params {
    self.params.clone()
  }
}

/// G2++ calibrator: `(a, b, σ, η, ρ)` of the two-factor Gaussian model against
/// a swaption grid on the two-dimensional trinomial tree.
#[derive(Debug, Clone)]
pub struct G2ppSwaptionCalibrator<'a> {
  pub quotes: &'a [SwaptionQuote],
  pub curve: &'a DiscountCurve<f64>,
  pub notional: f64,
  pub initial_rate: f64,
  /// Tree levels per year.
  pub steps_per_year: usize,
  pub initial_guess: Option<[f64; 5]>,
  pub max_iters: u64,
  pub sd_tolerance: f64,
  /// Optional Tikhonov pull of `(a, b, σ, η, ρ)` toward an anchor.
  pub regularization: Option<Regularization>,
}

impl<'a> G2ppSwaptionCalibrator<'a> {
  pub fn new(
    quotes: &'a [SwaptionQuote],
    curve: &'a DiscountCurve<f64>,
    notional: f64,
    initial_rate: f64,
    steps_per_year: usize,
  ) -> Self {
    Self {
      quotes,
      curve,
      notional,
      initial_rate,
      steps_per_year,
      initial_guess: None,
      max_iters: 400,
      sd_tolerance: 1e-10,
      regularization: None,
    }
  }

  /// Overrides the Nelder–Mead iteration cap.
  pub fn with_max_iters(mut self, max_iters: u64) -> Self {
    self.max_iters = max_iters;
    self
  }

  /// Adds a Tikhonov pull toward `regularization.anchor` in the order
  /// `(a, b, σ, η, ρ)`.
  pub fn with_regularization(mut self, regularization: Regularization) -> Self {
    assert_eq!(
      regularization.dimension(),
      5,
      "G2++ regularisation needs five anchors"
    );
    self.regularization = Some(regularization);
    self
  }

  fn cost(&self) -> G2ppCost {
    G2ppCost {
      quotes: self.quotes.to_vec(),
      curve_points: serialize_curve(self.curve),
      notional: self.notional,
      initial_rate: self.initial_rate,
      steps_per_year: self.steps_per_year,
      regularization: self.regularization.clone(),
    }
  }

  fn solve(&self) -> G2ppCalibrationResult {
    let problem = self.cost();
    let x0 = self.initial_guess.unwrap_or([0.5, 0.05, 0.01, 0.005, -0.5]);
    let mut simplex = vec![x0.to_vec()];
    for i in 0..5 {
      let mut v = x0.to_vec();
      v[i] = if i == 4 {
        (x0[4] * 0.5).clamp(-0.9, 0.9)
      } else {
        x0[i] * 1.5
      };
      simplex.push(v);
    }
    let (best, converged) =
      run_nelder_mead(problem.clone(), simplex, self.max_iters, self.sd_tolerance);
    let params = G2ppCost::params(&best);
    let (model_prices, market_prices) = problem.price_series(&params);
    G2ppCalibrationResult {
      params,
      rmse: rmse(&model_prices, &market_prices),
      converged,
      model_prices,
      market_prices,
    }
  }
}

impl<'a> crate::traits::Calibrator for G2ppSwaptionCalibrator<'a> {
  type InitialGuess = [f64; 5];
  type Params = G2ppParams;
  type Output = G2ppCalibrationResult;
  type Error = anyhow::Error;

  fn calibrate(&self, initial: Option<Self::InitialGuess>) -> Result<Self::Output, Self::Error> {
    let mut this = self.clone();
    if let Some(guess) = initial {
      this.initial_guess = Some(guess);
    }
    Ok(this.solve())
  }
}

#[derive(Clone)]
struct G2ppCost {
  quotes: Vec<SwaptionQuote>,
  curve_points: CurveSnapshot,
  notional: f64,
  initial_rate: f64,
  steps_per_year: usize,
  regularization: Option<Regularization>,
}

impl G2ppCost {
  /// Positive reversions and volatilities, correlation clamped inside
  /// `(−1, 1)`.
  fn params(x: &[f64]) -> G2ppParams {
    G2ppParams {
      mean_reversion_x: x[0].abs().max(1e-6),
      mean_reversion_y: x[1].abs().max(1e-6),
      sigma_x: x[2].abs().max(1e-6),
      sigma_y: x[3].abs().max(1e-6),
      rho: x[4].clamp(-0.999, 0.999),
    }
  }

  fn price_series(&self, p: &G2ppParams) -> (Vec<f64>, Vec<f64>) {
    let curve = self.curve_points.rebuild();
    let model = G2ppTreeModel::new(
      0.0,
      0.0,
      self.initial_rate,
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
      let w = quote.weight.unwrap_or(1.0);
      model_prices.push(w * swaption.price_on_g2pp(&tree));
      market_prices.push(w * market.market_price);
    }
    (model_prices, market_prices)
  }
}

impl CostFunction for G2ppCost {
  type Param = Vec<f64>;
  type Output = f64;
  fn cost(&self, x: &Self::Param) -> Result<f64, argmin::core::Error> {
    let p = Self::params(x);
    let (model, market) = self.price_series(&p);
    let penalty = self.regularization.as_ref().map_or(0.0, |reg| {
      reg.penalty(&[
        p.mean_reversion_x,
        p.mean_reversion_y,
        p.sigma_x,
        p.sigma_y,
        p.rho,
      ])
    });
    Ok(
      model
        .iter()
        .zip(&market)
        .map(|(m, q)| (m - q).powi(2))
        .sum::<f64>()
        + penalty,
    )
  }
}

#[cfg(test)]
mod tests;
