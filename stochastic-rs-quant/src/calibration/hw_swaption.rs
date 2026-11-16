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

use std::convert::Infallible;

use basin::CostFunction;

use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationOptions;
use crate::calibration::Regularization;
use crate::calibration::nelder_mead_fit;
use crate::calibration::quote::check_finite_start;
use crate::curves::DiscountCurve;
use crate::instruments::option::caplet::black_forward_caplet;
use crate::instruments::option::jamshidian::price_jamshidian_hull_white;
use crate::instruments::option::types::SwaptionDirection;
use crate::lattice::short_rate::HullWhiteTreeModel;
use crate::traits::ToShortRateModel;

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

/// `InsufficientData` for no quote; `InvalidQuote` for an expiry, tenor or accrual not finite and
/// `> 0`, a vol or weight not finite and `≥ 0`, or a negative forward swap rate on `curve`.
pub(crate) fn validate_swaption_quotes(
  quotes: &[SwaptionQuote],
  curve: &DiscountCurve<f64>,
) -> Result<(), QuantError> {
  if quotes.is_empty() {
    return Err(QuantError::InsufficientData {
      what: "swaption quotes",
      needed: 1,
      got: 0,
    });
  }
  for (index, quote) in quotes.iter().enumerate() {
    let fields = [
      ("expiry", quote.expiry, true),
      ("tenor", quote.tenor, true),
      ("fixed_accrual", quote.fixed_accrual, true),
      ("black_vol", quote.black_vol, false),
      ("weight", quote.weight.unwrap_or(1.0), false),
    ];
    for (name, value, positive) in fields {
      let ok = value.is_finite() && if positive { value > 0.0 } else { value >= 0.0 };
      if !ok {
        let bound = if positive { "> 0.0" } else { ">= 0.0" };
        let reason = format!(
          "{name} must satisfy `{name}.is_finite() && {name} {bound}`, got {name} = {value:?}"
        );
        return Err(QuantError::InvalidQuote { index, reason });
      }
    }
    let fair_rate = fair_rate(quote, curve);
    if fair_rate < 0.0 {
      let reason = format!(
        "fair_rate must satisfy `fair_rate >= 0.0`, as Black-76 prices no negative forward, got fair_rate = {fair_rate:?}"
      );
      return Err(QuantError::InvalidQuote { index, reason });
    }
  }
  Ok(())
}

/// A quote's forward swap rate on the schedule both swaption pricers build; its sign is theirs.
fn fair_rate(quote: &SwaptionQuote, curve: &DiscountCurve<f64>) -> f64 {
  let n_payments = ((quote.tenor * (1.0 / quote.fixed_accrual).round()).round() as usize).max(1);
  let accrual = quote.tenor / n_payments as f64;
  let annuity = (1..=n_payments)
    .map(|k| curve.discount_factor(quote.expiry + accrual * k as f64) * accrual)
    .sum::<f64>();
  let p_end = curve.discount_factor(quote.expiry + accrual * n_payments as f64);
  (curve.discount_factor(quote.expiry) - p_end) / annuity
}

/// `Σ (wᵢ·modelᵢ − wᵢ·marketᵢ)²`: the swaption objective, while `rmse()` stays on the raw prices.
pub(crate) fn weighted_sse(quotes: &[SwaptionQuote], market: &[f64], model: &[f64]) -> f64 {
  quotes
    .iter()
    .zip(market.iter().zip(model))
    .map(|(quote, (q, m))| {
      let w = quote.weight.unwrap_or(1.0);
      (w * m - w * q).powi(2)
    })
    .sum()
}

/// Hull–White parameters; the tree inputs only shape the model `to_short_rate_model` returns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HullWhiteParams {
  /// Mean reversion speed $a$.
  pub mean_reversion: f64,
  /// Short-rate volatility $\sigma$.
  pub sigma: f64,
  /// Time-0 short rate of the tree model; fixed during the fit.
  pub initial_rate: f64,
  /// Constant drift offset of the tree model; fixed during the fit.
  pub theta: f64,
}

impl HullWhiteParams {
  fn fields(&self) -> [(&'static str, f64); 4] {
    [
      ("mean_reversion", self.mean_reversion),
      ("sigma", self.sigma),
      ("initial_rate", self.initial_rate),
      ("theta", self.theta),
    ]
  }
}

impl ToShortRateModel for Calibrated<HullWhiteParams> {
  type Model = HullWhiteTreeModel<f64>;

  fn to_short_rate_model(&self) -> Self::Model {
    let p = &self.params;
    HullWhiteTreeModel::new(p.initial_rate, p.mean_reversion, p.theta, p.sigma)
  }
}

/// Hull–White `(a, σ)` against a swaption grid priced by Jamshidian's decomposition on `curve`.
#[derive(Debug, Clone)]
pub struct HullWhiteSwaptionCalibrator<'a> {
  pub quotes: &'a [SwaptionQuote],
  pub curve: &'a DiscountCurve<f64>,
  /// Notional used consistently for all quotes.
  pub notional: f64,
  pub start: HullWhiteParams,
  /// Nelder–Mead: `max_iterations` (default 400), `tolerance` its spread; pull `(a, σ)`.
  pub options: CalibrationOptions,
}

impl<'a> HullWhiteSwaptionCalibrator<'a> {
  /// `start` carries the tree inputs (`initial_rate`, `theta`) and the first `(a, σ)`; panics
  /// unless `notional > 0` and the start is finite.
  pub fn new(
    quotes: &'a [SwaptionQuote],
    curve: &'a DiscountCurve<f64>,
    notional: f64,
    start: HullWhiteParams,
  ) -> Self {
    assert!(
      notional.is_finite() && notional > 0.0,
      "notional must satisfy `notional.is_finite() && notional > 0.0`, got notional = {notional:?}"
    );
    check_finite_start(&start.fields()).unwrap_or_else(|e| panic!("{e}"));
    Self {
      quotes,
      curve,
      notional,
      start,
      options: CalibrationOptions::default(),
    }
  }

  pub fn with_options(mut self, options: CalibrationOptions) -> Self {
    self.options = options;
    self
  }
}

impl crate::traits::Calibrator for HullWhiteSwaptionCalibrator<'_> {
  type Params = HullWhiteParams;
  type Output = Calibrated<HullWhiteParams>;
  type Error = QuantError;

  /// A start replaces `self.start` whole, tree inputs included.
  fn calibrate(&self, initial: Option<HullWhiteParams>) -> Result<Self::Output, QuantError> {
    let start = initial.unwrap_or(self.start);
    check_finite_start(&start.fields())?;
    validate_swaption_quotes(self.quotes, self.curve)?;
    let problem = HullWhiteCost {
      quotes: self.quotes.to_vec(),
      curve_points: serialize_curve(self.curve),
      notional: self.notional,
      regularization: self.options.regularization_for(2)?,
    };
    let (a0, s0) = (start.mean_reversion, start.sigma);
    let simplex = vec![vec![a0, s0], vec![a0 * 1.5, s0], vec![a0, s0 * 1.5]];
    Ok(nelder_mead_fit(
      problem.clone(),
      simplex,
      &self.options,
      400,
      |x| {
        let (mean_reversion, sigma) = (x[0].abs(), x[1].abs());
        let (market, model) = problem.price_series(mean_reversion, sigma);
        let params = HullWhiteParams {
          mean_reversion,
          sigma,
          ..start
        };
        (params, market, model)
      },
    ))
  }
}

#[derive(Clone)]
struct HullWhiteCost {
  quotes: Vec<SwaptionQuote>,
  curve_points: CurveSnapshot,
  notional: f64,
  regularization: Option<Regularization>,
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

      model_prices.push(model_price);
      market_prices.push(market_price);
    }
    (market_prices, model_prices)
  }
}

impl CostFunction for HullWhiteCost {
  type Param = Vec<f64>;
  type Output = f64;
  type Error = Infallible;

  fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
    let a = x[0].abs().max(1e-6);
    let sigma = x[1].abs().max(1e-6);
    let (market_prices, model_prices) = self.price_series(a, sigma);
    let loss = weighted_sse(&self.quotes, &market_prices, &model_prices);
    let penalty = self
      .regularization
      .as_ref()
      .map_or(0.0, |reg| reg.penalty(&[a, sigma]));
    Ok(loss + penalty)
  }
}

#[derive(Debug, Clone)]
pub(crate) struct CurveSnapshot {
  times: Vec<f64>,
  rates: Vec<f64>,
  method: crate::curves::InterpolationMethod,
}

impl CurveSnapshot {
  pub(crate) fn rebuild(&self) -> DiscountCurve<f64> {
    let times = ndarray::Array1::from(self.times.clone());
    let rates = ndarray::Array1::from(self.rates.clone());
    DiscountCurve::from_zero_rates(&times, &rates, self.method)
  }
}

pub(crate) fn serialize_curve(curve: &DiscountCurve<f64>) -> CurveSnapshot {
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

#[cfg(test)]
mod tests {
  use ndarray::Array1;

  use super::*;
  use crate::curves::InterpolationMethod;
  use crate::traits::CalibrationResult;
  use crate::traits::Calibrator;

  const START: HullWhiteParams = HullWhiteParams {
    mean_reversion: 0.1,
    sigma: 0.01,
    initial_rate: 0.03,
    theta: 0.002,
  };

  fn curve() -> DiscountCurve<f64> {
    DiscountCurve::from_zero_rates(
      &Array1::from_vec(vec![0.5, 1.0, 5.0, 10.0]),
      &Array1::from_vec(vec![0.03; 4]),
      InterpolationMethod::LogLinearOnDiscountFactors,
    )
  }

  fn quotes() -> Vec<SwaptionQuote> {
    [(1.0, 2.0, 0.22), (2.0, 3.0, 0.20)]
      .into_iter()
      .map(|(expiry, tenor, black_vol)| SwaptionQuote {
        expiry,
        tenor,
        black_vol,
        fixed_accrual: 0.5,
        direction: SwaptionDirection::Payer,
        weight: None,
      })
      .collect()
  }

  /// A heavy pull on the mean reversion pins it at the anchor; the volatility
  /// stays free to fit the quotes.
  #[test]
  fn anchor_on_mean_reversion_pins_it() {
    let (curve, quotes) = (curve(), quotes());
    let plain = HullWhiteSwaptionCalibrator::new(&quotes, &curve, 1.0, START)
      .calibrate(None)
      .expect("calibration runs");
    let pull = Regularization::new(vec![0.3, 0.01], vec![1e6, 0.0]);
    let pulled = HullWhiteSwaptionCalibrator::new(&quotes, &curve, 1.0, START)
      .with_options(CalibrationOptions::default().with_regularization(pull))
      .calibrate(None)
      .expect("calibration runs");
    let a = pulled.params.mean_reversion;
    assert!((a - 0.3).abs() < 1e-2, "a {a}");
    assert!(pulled.params.sigma > 0.0 && plain.params.sigma > 0.0);
  }

  #[test]
  fn the_result_carries_the_tree_inputs_it_was_given() {
    let (curve, quotes) = (curve(), quotes());
    let result = HullWhiteSwaptionCalibrator::new(&quotes, &curve, 1.0, START)
      .calibrate(None)
      .expect("calibration runs");
    let model = result.to_short_rate_model();
    assert_eq!((model.initial_rate, model.theta), (0.03, 0.002));
    assert_eq!(model.sigma, result.params.sigma);
  }

  #[test]
  fn the_recorded_trail_ends_at_the_fit() {
    let (curve, quotes) = (curve(), quotes());
    let calibrator = HullWhiteSwaptionCalibrator::new(&quotes, &curve, 1.0, START);
    let plain = calibrator.calibrate(None).unwrap();
    let recorded = calibrator
      .clone()
      .with_options(CalibrationOptions::default().with_record_history(true))
      .calibrate(None)
      .unwrap();
    assert!(plain.history().is_empty());
    assert_eq!(recorded.params, plain.params);
    assert_eq!(recorded.history.len(), recorded.iterations.unwrap() + 1);
    let last = recorded.history.last().unwrap();
    assert_eq!(
      (last.iteration, last.params),
      (recorded.iterations.unwrap(), recorded.params)
    );
    assert_eq!(last.loss.get(crate::LossMetric::Rmse), recorded.rmse());
  }

  #[test]
  fn rmse_and_max_error_are_unweighted_price_errors() {
    let (curve, mut quotes) = (curve(), quotes());
    quotes[0].weight = Some(4.0);
    quotes[1].weight = Some(0.25);
    let fit = HullWhiteSwaptionCalibrator::new(&quotes, &curve, 1.0, START)
      .calibrate(None)
      .unwrap();
    let unweighted: Vec<_> = quotes
      .iter()
      .map(|quote| SwaptionQuote {
        weight: None,
        ..quote.clone()
      })
      .collect();
    let cost = HullWhiteCost {
      quotes: unweighted,
      curve_points: serialize_curve(&curve),
      notional: 1.0,
      regularization: None,
    };
    let (market, model) = cost.price_series(fit.params.mean_reversion, fit.params.sigma);
    assert_eq!(fit.rmse(), crate::loss::rmse(&market, &model));
    let worst = market.iter().zip(&model).map(|(m, q)| (m - q).abs());
    assert_eq!(fit.max_error, Some(worst.fold(0.0, f64::max)));
  }

  #[test]
  fn bad_data_is_an_error() {
    let (curve, mut quotes) = (curve(), quotes());
    let fit = |quotes: &[SwaptionQuote], start: Option<HullWhiteParams>, options| {
      HullWhiteSwaptionCalibrator::new(quotes, &curve, 1.0, START)
        .with_options(options)
        .calibrate(start)
        .unwrap_err()
    };
    let defaults = CalibrationOptions::default;
    let empty = fit(&[], None, defaults());
    assert!(
      matches!(empty, QuantError::InsufficientData { got: 0, .. }),
      "{empty}"
    );
    let nan_start = Some(HullWhiteParams {
      sigma: f64::NAN,
      ..START
    });
    let start = fit(&quotes, nan_start, defaults());
    assert!(
      matches!(start, QuantError::InvalidInput { what: "start", .. }),
      "{start}"
    );
    let pull = Regularization::uniform(vec![0.1, 0.01, 0.0], 1.0);
    let reg = fit(&quotes, None, defaults().with_regularization(pull));
    assert!(
      matches!(
        reg,
        QuantError::DimensionMismatch {
          expected: 2,
          got: 3,
          ..
        }
      ),
      "{reg}"
    );
    quotes[1].black_vol = f64::NAN;
    let quote = fit(&quotes, None, defaults());
    assert!(
      matches!(quote, QuantError::InvalidQuote { index: 1, .. }),
      "{quote}"
    );
  }

  /// The forward turns negative past three years, so the 2y × 3y swap rate is about −3 %.
  #[test]
  fn a_quote_whose_swap_rate_is_negative_has_no_black_price_and_is_refused() {
    let curve = DiscountCurve::from_zero_rates(
      &Array1::from_vec(vec![1.0, 3.0, 5.0]),
      &Array1::from_vec(vec![0.02, 0.02, -0.01]),
      InterpolationMethod::LogLinearOnDiscountFactors,
    );
    let quotes = quotes();
    let err = HullWhiteSwaptionCalibrator::new(&quotes, &curve, 1.0, START)
      .calibrate(None)
      .unwrap_err();
    assert!(
      matches!(&err, QuantError::InvalidQuote { index: 1, reason } if reason.contains("fair_rate")),
      "{err}"
    );
  }

  #[test]
  #[should_panic(expected = "invalid start: theta must satisfy `theta.is_finite()`")]
  fn a_non_finite_start_in_new_panics() {
    let (curve, quotes) = (curve(), quotes());
    let start = HullWhiteParams {
      theta: f64::INFINITY,
      ..START
    };
    HullWhiteSwaptionCalibrator::new(&quotes, &curve, 1.0, start);
  }
}
