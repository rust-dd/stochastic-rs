//! # Heston Stochastic Correlation Calibration
//!
//! Calibrates the 9-parameter HSCM model to market option prices using
//! SLSQP (Sequential Least Squares Programming) with bounds — the same
//! optimizer as the Python reference implementation.
//!
//! Parameters: \[κ_v, θ_v, σ_v, v₀, κ_r, μ_r, σ_r, ρ₀, ρ₂\]
//!
//! ## References
//! - Teng, Ehrhardt & Günther (2016)
//! - Tanaś, R. — <https://github.com/tanasr/HestonStochCorr>

use crate::LossMetric;
use crate::OptionQuote;
use crate::OptionType;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationOptions;
use crate::calibration::quote::assert_spot_and_rates;
use crate::calibration::quote::check_finite_start;
use crate::calibration::quote::validate_quotes;
use crate::pricing::heston_stoch_corr::HestonStochCorrPricer;
use crate::traits::ToModel;

/// Parameter bounds for HSCM calibration (matches Python repo).
pub(crate) const BOUNDS: [(f64, f64); 9] = [
  (0.01, 10.0),  // kappa_v
  (0.001, 1.0),  // theta_v
  (0.01, 2.0),   // sigma_v
  (0.001, 0.5),  // v0
  (0.01, 20.0),  // kappa_r
  (-0.99, 0.99), // mu_r
  (0.01, 2.0),   // sigma_r
  (-0.99, 0.99), // rho0
  (-0.99, 0.99), // rho2
];

/// A mild-skew, short-tenor SPX-style start in `HscmParams` order.
const DEFAULT_START: [f64; 9] = [2.0, 0.04, 0.3, 0.04, 5.0, -0.5, 0.2, -0.7, 0.3];

const NAMES: [&str; 9] = [
  "kappa_v", "theta_v", "sigma_v", "v0", "kappa_r", "mu_r", "sigma_r", "rho0", "rho2",
];

/// Calibrated parameter set for the Heston-stochastic-correlation model.
#[derive(Clone, Debug, PartialEq)]
pub struct HscmParams {
  pub kappa_v: f64,
  pub theta_v: f64,
  pub sigma_v: f64,
  pub v0: f64,
  pub kappa_r: f64,
  pub mu_r: f64,
  pub sigma_r: f64,
  pub rho0: f64,
  pub rho2: f64,
}

impl HscmParams {
  fn values(&self) -> [f64; 9] {
    [
      self.kappa_v,
      self.theta_v,
      self.sigma_v,
      self.v0,
      self.kappa_r,
      self.mu_r,
      self.sigma_r,
      self.rho0,
      self.rho2,
    ]
  }

  fn from_values(x: &[f64]) -> Self {
    Self {
      kappa_v: x[0],
      theta_v: x[1],
      sigma_v: x[2],
      v0: x[3],
      kappa_r: x[4],
      mu_r: x[5],
      sigma_r: x[6],
      rho0: x[7],
      rho2: x[8],
    }
  }
}

impl From<HscmParams> for HestonStochCorrPricer {
  fn from(p: HscmParams) -> Self {
    Self {
      v0: p.v0,
      kappa_v: p.kappa_v,
      theta_v: p.theta_v,
      sigma_v: p.sigma_v,
      rho0: p.rho0,
      kappa_r: p.kappa_r,
      mu_r: p.mu_r,
      sigma_r: p.sigma_r,
      rho2: p.rho2,
    }
  }
}

impl<D> ToModel for Calibrated<HscmParams, D> {
  type Model = HestonStochCorrPricer;

  fn to_model(&self) -> HestonStochCorrPricer {
    HestonStochCorrPricer::from(self.params.clone())
  }
}

/// The objective's market: spot, the one rate pair and the quotes.
#[derive(Clone)]
struct CalibData {
  s: f64,
  r: f64,
  q: f64,
  quotes: Vec<OptionQuote>,
}

impl CalibData {
  fn price(&self, x: &[f64], quote: &OptionQuote) -> f64 {
    let model = HestonStochCorrPricer::new(x[3], x[0], x[1], x[2], x[7], x[4], x[5], x[6], x[8]);
    let (k, tau) = (quote.strike, quote.tau);
    match quote.option_type {
      OptionType::Call => model.price_call_carr_madan(self.s, k, self.r, self.q, tau),
      OptionType::Put => model.call_put(self.s, k, self.r, self.q, tau).1,
    }
  }

  /// The sum of squared relative price errors.
  fn sse(&self, x: &[f64]) -> f64 {
    let mut obj = 0.0;
    for quote in &self.quotes {
      let err = (self.price(x, quote) - quote.price) / quote.price.max(1e-6);
      obj += err * err;
    }
    obj
  }
}

/// SLSQP objective with a forward-difference gradient.
fn slsqp_objective(x: &[f64], gradient: Option<&mut [f64]>, data: &mut CalibData) -> f64 {
  if let Some(g) = gradient {
    let h = 1e-5;
    let f0 = data.sse(x);
    for i in 0..x.len() {
      let mut xp = x.to_vec();
      xp[i] += h;
      g[i] = (data.sse(&xp) - f0) / h;
    }
  }
  data.sse(x)
}

/// HSCM fit to vanilla quotes by SLSQP inside `BOUNDS`, on the squared relative price errors.
#[derive(Clone, Debug)]
pub struct HscmCalibrator {
  pub s: f64,
  pub r: f64,
  pub q: f64,
  pub quotes: Vec<OptionQuote>,
  /// `None` starts from (2, 0.04, 0.3, 0.04, 5, −0.5, 0.2, −0.7, 0.3); a start is clamped to
  /// `BOUNDS`.
  pub start: Option<HscmParams>,
  /// SLSQP: `max_iterations` caps its evaluations (default 500); `regularization` is `Unsupported`.
  pub options: CalibrationOptions,
}

impl HscmCalibrator {
  /// Panics unless `s` is finite and positive and `r`, `q` finite; `calibrate` checks the quotes.
  pub fn new(s: f64, r: f64, q: f64, quotes: Vec<OptionQuote>) -> Self {
    assert_spot_and_rates(s, r, q);
    Self {
      s,
      r,
      q,
      quotes,
      start: None,
      options: CalibrationOptions::default(),
    }
  }

  /// Panics unless every value of `start` is finite.
  pub fn with_start(mut self, start: HscmParams) -> Self {
    check_start(&start).unwrap_or_else(|e| panic!("{e}"));
    self.start = Some(start);
    self
  }

  pub fn with_options(mut self, options: CalibrationOptions) -> Self {
    self.options = options;
    self
  }
}

fn check_start(start: &HscmParams) -> Result<(), QuantError> {
  let fields = NAMES.into_iter().zip(start.values()).collect::<Vec<_>>();
  check_finite_start(&fields)
}

impl crate::traits::Calibrator for HscmCalibrator {
  type Params = HscmParams;
  type Output = Calibrated<HscmParams>;
  type Error = QuantError;

  /// A start replaces `self.start`; `InvalidInput` unless every value of the one used is finite,
  /// `Optimizer` when the objective at it is not finite.
  fn calibrate(&self, initial: Option<HscmParams>) -> Result<Calibrated<HscmParams>, QuantError> {
    validate_quotes(&self.quotes)?;
    if self.options.regularization.is_some() {
      return Err(QuantError::Unsupported {
        what: "regularization",
      });
    }
    let start = initial.or_else(|| self.start.clone());
    if let Some(start) = &start {
      check_start(start)?;
    }
    let x_init = start
      .map_or(DEFAULT_START, |p| p.values())
      .iter()
      .zip(BOUNDS)
      .map(|(&v, (lower, upper))| v.clamp(lower, upper))
      .collect::<Vec<_>>();
    let data = CalibData {
      s: self.s,
      r: self.r,
      q: self.q,
      quotes: self.quotes.clone(),
    };
    let start_objective = data.sse(&x_init);
    if !start_objective.is_finite() {
      return Err(QuantError::Optimizer {
        what: "slsqp fit",
        reason: format!("the objective at the start is {start_objective:?}"),
      });
    }
    let cons: Vec<&dyn slsqp::Func<CalibData>> = vec![];
    let (x, final_objective, ok) = match slsqp::minimize(
      slsqp_objective,
      &x_init,
      &BOUNDS,
      &cons,
      data.clone(),
      self.options.max_iterations.unwrap_or(500),
      None,
    ) {
      Ok((_status, x_opt, fval)) => (x_opt, fval, true),
      Err((_status, x_opt, fval)) => (x_opt, fval, false),
    };
    let market = self
      .quotes
      .iter()
      .map(|quote| quote.price)
      .collect::<Vec<_>>();
    let model = self
      .quotes
      .iter()
      .map(|quote| data.price(&x, quote))
      .collect::<Vec<_>>();
    let params = HscmParams::from_values(&x);
    let mut fit = Calibrated::repriced(params, &market, &model, self.options.loss_metrics, ok);
    // An SLSQP success that reprices to a non-finite error is not a fit.
    fit.converged &= fit.loss.get(LossMetric::Rmse).is_finite();
    Ok(fit.with_message(format!("slsqp objective {final_objective:e}")))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::Calibrator;

  fn quotes(x: &[f64]) -> Vec<OptionQuote> {
    let data = CalibData {
      s: 100.0,
      r: 0.03,
      q: 0.0,
      quotes: Vec::new(),
    };
    [95.0, 100.0, 105.0]
      .map(|k| {
        let call = OptionQuote::new(k, 0.25, 0.0, OptionType::Call);
        OptionQuote::new(k, 0.25, data.price(x, &call), OptionType::Call)
      })
      .to_vec()
  }

  #[test]
  #[ignore = "slow: HSCM Carr-Madan FFT × SLSQP iterations. Run with --ignored."]
  fn calibrate_recovers_synthetic_prices() {
    let truth = [2.0, 0.04, 0.3, 0.04, 5.0, -0.5, 0.2, -0.7, 0.3];
    let guess = [1.5, 0.05, 0.4, 0.05, 4.0, -0.3, 0.3, -0.5, 0.2];
    let calibrator = HscmCalibrator::new(100.0, 0.03, 0.0, quotes(&truth))
      .with_start(HscmParams::from_values(&guess))
      .with_options(CalibrationOptions::default().with_max_iterations(200));
    let initial = CalibData {
      s: 100.0,
      r: 0.03,
      q: 0.0,
      quotes: calibrator.quotes.clone(),
    }
    .sse(&guess);
    let fit = calibrator.calibrate(None).unwrap();
    let message = fit.message.as_deref().unwrap();
    let final_objective = message
      .strip_prefix("slsqp objective ")
      .unwrap()
      .parse::<f64>()
      .unwrap();
    assert!(
      final_objective < initial * 0.5,
      "final {final_objective} initial {initial}"
    );
    assert!(fit.loss.get(LossMetric::Rmse) < 1e-2, "{:?}", fit.loss);
  }

  #[test]
  fn a_pull_is_unsupported_and_a_start_must_be_finite() {
    let calibrator = HscmCalibrator::new(100.0, 0.03, 0.0, quotes(&DEFAULT_START));
    let pull = crate::calibration::Regularization::uniform(vec![0.0; 9], 1.0);
    let err = calibrator
      .clone()
      .with_options(CalibrationOptions::default().with_regularization(pull))
      .calibrate(None)
      .unwrap_err();
    assert_eq!(
      err,
      QuantError::Unsupported {
        what: "regularization"
      }
    );
    let start = HscmParams {
      rho2: f64::NAN,
      ..HscmParams::from_values(&DEFAULT_START)
    };
    let err = calibrator.calibrate(Some(start)).unwrap_err();
    assert!(
      matches!(&err, QuantError::InvalidInput { what: "start", reason } if reason.starts_with("rho2")),
      "{err}"
    );
  }
}
