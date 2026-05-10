//! Hull-White calibration to a European swaption volatility grid.
//!
//! Minimizes the sum of squared differences between Jamshidian Hull-White
//! prices and Black-76 ATM prices computed from market volatilities.
//!
//! $$
//! (\hat a,\hat\sigma)=\arg\min_{a,\sigma}
//!   \sum_{i} w_i\bigl(V^{HW}_i(a,\sigma)-V^{B76}_i(\sigma^{mkt}_i)\bigr)^2
//! $$
//!
//! Reference: Brigo & Mercurio, "Interest Rate Models — Theory and Practice",
//! Springer 2nd ed. (2006), §3.3.2 & §3.11.

use argmin::core::CostFunction;
use argmin::core::Executor;
use argmin::core::State;
use argmin::solver::neldermead::NelderMead;

use crate::curves::DiscountCurve;
use crate::instruments::option::caplet::black_forward_caplet;
use crate::instruments::option::jamshidian::price_jamshidian_hull_white;
use crate::instruments::option::types::SwaptionDirection;

/// Single European swaption quote entered as a market Black-76 volatility.
#[derive(Debug, Clone)]
pub struct SwaptionQuote {
  /// Time from valuation to exercise in years.
  pub expiry: f64,
  /// Swap tenor in years (from exercise to final maturity).
  pub tenor: f64,
  /// Black-76 at-the-money-forward volatility.
  pub black_vol: f64,
  /// Fixed-leg accrual convention in years (e.g., 0.5 for semi-annual).
  pub fixed_accrual: f64,
  /// Payer / receiver direction.
  pub direction: SwaptionDirection,
  /// Optional quote weight; defaults to 1.0 if `None`.
  pub weight: Option<f64>,
}

/// Calibrated parameter set for the Hull-White short-rate model.
#[derive(Debug, Clone)]
pub struct HullWhiteParams {
  /// Mean reversion speed $a$.
  pub mean_reversion: f64,
  /// Short-rate volatility $\sigma$.
  pub sigma: f64,
}

/// Hull-White calibration result.
#[derive(Debug, Clone)]
pub struct HullWhiteCalibrationResult {
  /// Calibrated mean reversion $a$.
  pub mean_reversion: f64,
  /// Calibrated volatility $\sigma$.
  pub sigma: f64,
  /// Root-mean-square error on the quote grid.
  pub rmse: f64,
  /// True when the Nelder-Mead simplex reported convergence.
  pub converged: bool,
  /// Per-quote model prices in the order of the input quotes.
  pub model_prices: Vec<f64>,
  /// Per-quote market prices in the same order.
  pub market_prices: Vec<f64>,
}

impl HullWhiteCalibrationResult {
  /// Convert to a [`HullWhiteTreeModel`](crate::lattice::HullWhiteTreeModel)
  /// for short-rate / interest-rate option valuation via the lattice pipeline.
  ///
  /// Hull-White lattice models do **not** plug into the equity vol-surface
  /// pipeline (`ToModel` / `ModelPricer`); they consume a yield curve and a
  /// time grid and produce a discount tree. Use the lattice instruments
  /// (`Cap`, `Floor`, `BermudanSwaption`, …) for valuation.
  pub fn to_short_rate_model(
    &self,
    initial_rate: f64,
    theta: f64,
  ) -> crate::lattice::short_rate::HullWhiteTreeModel<f64> {
    crate::lattice::short_rate::HullWhiteTreeModel::new(
      initial_rate,
      self.mean_reversion,
      theta,
      self.sigma,
    )
  }
}

impl crate::traits::ToShortRateModel for HullWhiteCalibrationResult {
  type Model = crate::lattice::short_rate::HullWhiteTreeModel<f64>;
  fn to_short_rate_model(&self, initial_rate: f64, theta: f64) -> Self::Model {
    HullWhiteCalibrationResult::to_short_rate_model(self, initial_rate, theta)
  }
}

impl crate::traits::CalibrationResult for HullWhiteCalibrationResult {
  type Params = HullWhiteParams;
  fn rmse(&self) -> f64 {
    self.rmse
  }
  fn converged(&self) -> bool {
    self.converged
  }
  fn params(&self) -> Self::Params {
    HullWhiteParams {
      mean_reversion: self.mean_reversion,
      sigma: self.sigma,
    }
  }
}

impl<'a> crate::traits::Calibrator for HullWhiteSwaptionCalibrator<'a> {
  type InitialGuess = (f64, f64);
  type Params = HullWhiteParams;
  type Output = HullWhiteCalibrationResult;
  type Error = anyhow::Error;

  fn calibrate(&self, initial: Option<Self::InitialGuess>) -> Result<Self::Output, Self::Error> {
    let mut this = self.clone();
    if let Some(guess) = initial {
      this.initial_guess = Some(guess);
    }
    Ok(this.solve())
  }
}

/// Hull-White calibrator.
#[derive(Debug, Clone)]
pub struct HullWhiteSwaptionCalibrator<'a> {
  /// Market quotes.
  pub quotes: &'a [SwaptionQuote],
  /// Initial market discount curve.
  pub curve: &'a DiscountCurve<f64>,
  /// Notional used consistently for all quotes.
  pub notional: f64,
  /// Optional initial guess `(a, σ)`; defaults to `(0.05, 0.01)`.
  pub initial_guess: Option<(f64, f64)>,
  /// Maximum Nelder-Mead iterations.
  pub max_iters: u64,
  /// Convergence tolerance on simplex standard deviation.
  pub sd_tolerance: f64,
}

impl<'a> HullWhiteSwaptionCalibrator<'a> {
  /// Construct a calibrator with sensible defaults.
  pub fn new(quotes: &'a [SwaptionQuote], curve: &'a DiscountCurve<f64>, notional: f64) -> Self {
    Self {
      quotes,
      curve,
      notional,
      initial_guess: None,
      max_iters: 400,
      sd_tolerance: 1e-10,
    }
  }

  fn solve(&self) -> HullWhiteCalibrationResult {
    let problem = HullWhiteCost {
      quotes: self.quotes.to_vec(),
      curve_points: serialize_curve(self.curve),
      notional: self.notional,
    };
    let (a0, s0) = self.initial_guess.unwrap_or((0.05, 0.01));
    let simplex = vec![vec![a0, s0], vec![a0 * 1.5, s0], vec![a0, s0 * 1.5]];

    let mut converged = true;
    let best = match NelderMead::new(simplex.clone()).with_sd_tolerance(self.sd_tolerance) {
      Ok(solver) => match Executor::new(problem.clone(), solver)
        .configure(|s| s.max_iters(self.max_iters))
        .run()
      {
        Ok(res) => res
          .state
          .get_best_param()
          .cloned()
          .unwrap_or_else(|| simplex[0].clone()),
        Err(_) => {
          converged = false;
          simplex[0].clone()
        }
      },
      Err(_) => {
        converged = false;
        simplex[0].clone()
      }
    };

    let a_hat = best[0].abs();
    let sigma_hat = best[1].abs();
    let (model_prices, market_prices) = problem.price_series(a_hat, sigma_hat);
    let residual_sq: f64 = model_prices
      .iter()
      .zip(market_prices.iter())
      .map(|(m, q)| (m - q).powi(2))
      .sum();
    let n = model_prices.len().max(1) as f64;
    let rmse = (residual_sq / n).sqrt();

    HullWhiteCalibrationResult {
      mean_reversion: a_hat,
      sigma: sigma_hat,
      rmse,
      converged,
      model_prices,
      market_prices,
    }
  }
}

#[derive(Clone)]
struct HullWhiteCost {
  quotes: Vec<SwaptionQuote>,
  curve_points: CurveSnapshot,
  notional: f64,
}

impl HullWhiteCost {
  fn curve(&self) -> DiscountCurve<f64> {
    self.curve_points.rebuild()
  }

  fn price_series(&self, a: f64, sigma: f64) -> (Vec<f64>, Vec<f64>) {
    let curve = self.curve();
    let mut model_prices = Vec::with_capacity(self.quotes.len());
    let mut market_prices = Vec::with_capacity(self.quotes.len());
    for quote in &self.quotes {
      let payments_per_year = (1.0 / quote.fixed_accrual).round();
      let n_payments = (quote.tenor * payments_per_year).round() as usize;
      let n_payments = n_payments.max(1);
      let accrual = quote.tenor / n_payments as f64;
      let coupon_times: Vec<f64> = (1..=n_payments)
        .map(|k| quote.expiry + accrual * k as f64)
        .collect();
      let accrual_factors = vec![accrual; n_payments];

      let annuity: f64 = coupon_times
        .iter()
        .map(|&t_i| curve.discount_factor(t_i) * accrual)
        .sum::<f64>()
        * self.notional;

      let p_exp = curve.discount_factor(quote.expiry);
      // `coupon_times` is built from `(1..=n_payments)` after `n_payments.max(1)`
      // above, so it is non-empty by construction. Use `expect` to encode the
      // invariant rather than blank `unwrap`.
      let p_end = curve.discount_factor(
        *coupon_times
          .last()
          .expect("coupon_times non-empty by n_payments.max(1) construction"),
      );
      let fair_rate = (p_exp - p_end) / (annuity / self.notional).max(1e-14);
      let forward_value = black_forward_caplet(fair_rate, fair_rate, quote.expiry, quote.black_vol);
      let market_price = annuity * forward_value;

      let model_price = price_jamshidian_hull_white(
        quote.direction,
        fair_rate,
        self.notional,
        quote.expiry,
        &coupon_times,
        &accrual_factors,
        a,
        sigma,
        &curve,
      );

      let w = quote.weight.unwrap_or(1.0);
      model_prices.push(w * model_price);
      market_prices.push(w * market_price);
    }
    (model_prices, market_prices)
  }
}

impl CostFunction for HullWhiteCost {
  type Param = Vec<f64>;
  type Output = f64;

  fn cost(&self, x: &Self::Param) -> Result<f64, argmin::core::Error> {
    let a = x[0].abs().max(1e-6);
    let sigma = x[1].abs().max(1e-6);
    let (model_prices, market_prices) = self.price_series(a, sigma);
    let loss: f64 = model_prices
      .iter()
      .zip(market_prices.iter())
      .map(|(m, q)| (m - q).powi(2))
      .sum();
    Ok(loss)
  }
}

#[derive(Debug, Clone)]
struct CurveSnapshot {
  times: Vec<f64>,
  rates: Vec<f64>,
  method: crate::curves::InterpolationMethod,
}

impl CurveSnapshot {
  fn rebuild(&self) -> DiscountCurve<f64> {
    let times = ndarray::Array1::from(self.times.clone());
    let rates = ndarray::Array1::from(self.rates.clone());
    DiscountCurve::from_zero_rates(&times, &rates, self.method)
  }
}

fn serialize_curve(curve: &DiscountCurve<f64>) -> CurveSnapshot {
  let points = curve.points();
  let times: Vec<f64> = points.iter().map(|p| p.time).collect();
  let rates: Vec<f64> = points
    .iter()
    .map(|p| {
      let t = p.time;
      if t <= 0.0 {
        0.0
      } else {
        -p.discount_factor.ln() / t
      }
    })
    .collect();
  CurveSnapshot {
    times,
    rates,
    method: curve.method(),
  }
}
