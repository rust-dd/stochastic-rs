//! # Bsm
//!
//! $$
//! C=S_0e^{(b-r)T}N(d_1)-Ke^{-rT}N(d_2),\quad d_{1,2}=\frac{\ln(S_0/K)+(b\pm\tfrac12\sigma^2)T}{\sigma\sqrt T}
//! $$
//!
//! Merton (1973), Bell J. Econ. Manag. Sci. 4(1), 141–183, DOI 10.2307/3003143.

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_distributions::RealExt;

use crate::OptionQuote;
use crate::OptionType;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationOptions;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::Trail;
use crate::calibration::least_squares::minimize;
use crate::calibration::quote::assert_spot_and_rates;
use crate::calibration::quote::check_finite_start;
use crate::calibration::quote::validate_quotes;
use crate::pricing::bsm::BsmCoc;
use crate::pricing::bsm::BsmPricer;
use crate::pricing::fourier::BsmFourier;
use crate::traits::ModelPricer;
use crate::traits::ToModel;

/// `SabrCalibrator`'s `ALPHA_MIN`, with no upper edge for the same reason: a fit to a distressed
/// surface can sit anywhere above it.
const V_MIN: f64 = 1e-6;

impl ToModel for Calibrated<BsmParams> {
  type Model = BsmFourier;

  fn to_model(&self) -> BsmFourier {
    BsmFourier {
      sigma: self.params.sigma,
    }
  }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BsmParams {
  pub sigma: f64,
}

impl BsmParams {
  /// Reflect `sigma` onto `[V_MIN, ∞)`, keeping the magnitude of a step that overshot below zero,
  /// where it would price a finite put-shaped call; a NaN iterate lands on `V_MIN`.
  pub fn project_in_place(&mut self) {
    self.sigma = self.sigma.abs().max(V_MIN);
  }

  /// [`project_in_place`](Self::project_in_place), by value.
  pub fn projected(mut self) -> Self {
    self.project_in_place();
    self
  }
}

impl From<BsmParams> for Array1<f64> {
  fn from(params: BsmParams) -> Self {
    Array1::from_vec(vec![params.sigma])
  }
}

impl From<Array1<f64>> for BsmParams {
  fn from(params: Array1<f64>) -> Self {
    BsmParams { sigma: params[0] }
  }
}

/// One volatility fitted to vanilla quotes by Levenberg–Marquardt on vega-weighted price
/// residuals, each quote priced at the carry `r − q`.
#[derive(Clone, Debug)]
pub struct BsmCalibrator {
  pub s: f64,
  pub r: f64,
  pub q: f64,
  pub quotes: Vec<OptionQuote>,
  /// Reflected onto `[1e-6, ∞)` before the first evaluation.
  pub start: BsmParams,
  /// Levenberg–Marquardt knobs; `regularization` pulls `σ`.
  pub options: CalibrationOptions,
}

impl BsmCalibrator {
  /// Panics unless `s` is finite and positive and `r`, `q` finite; `calibrate` checks the quotes.
  pub fn new(s: f64, r: f64, q: f64, quotes: Vec<OptionQuote>) -> Self {
    assert_spot_and_rates(s, r, q);
    Self {
      s,
      r,
      q,
      quotes,
      start: BsmParams { sigma: 0.2 },
      options: CalibrationOptions::default(),
    }
  }

  /// Panics unless `start.sigma` is finite; the start is stored reflected onto `[1e-6, ∞)`.
  pub fn with_start(mut self, start: BsmParams) -> Self {
    let sigma = start.sigma;
    assert!(
      sigma.is_finite(),
      "start.sigma must satisfy `start.sigma.is_finite()`, got start.sigma = {sigma:?}"
    );
    self.start = start.projected();
    self
  }

  pub fn with_options(mut self, options: CalibrationOptions) -> Self {
    self.options = options;
    self
  }
}

impl crate::traits::Calibrator for BsmCalibrator {
  type Params = BsmParams;
  type Output = Calibrated<BsmParams>;
  type Error = QuantError;

  /// A start replaces `self.start`; `InvalidInput` unless the one used is finite.
  fn calibrate(&self, initial: Option<BsmParams>) -> Result<Calibrated<BsmParams>, QuantError> {
    validate_quotes(&self.quotes)?;
    let start = initial.unwrap_or(self.start);
    check_finite_start(&[("sigma", start.sigma)])?;
    let lm = self.options.lm(false, 1)?;
    let (fitted, report) = minimize(BsmProblem::new(self, start.projected()), lm);
    let params = fitted.effective_params();
    let pricer = BsmPricer::new(params.sigma, BsmCoc::Merton1973);
    let model = self
      .quotes
      .iter()
      .map(|quote| {
        pricer.price_option(
          self.s,
          quote.strike,
          self.r,
          self.q,
          quote.tau,
          quote.option_type,
        )
      })
      .collect::<Vec<_>>();
    report.calibrated(
      params,
      &fitted.market,
      &model,
      self.options.loss_metrics,
      fitted.trail.into_entries(),
    )
  }
}

struct BsmProblem<'a> {
  calibrator: &'a BsmCalibrator,
  params: BsmParams,
  market: Vec<f64>,
  trail: Trail<BsmParams>,
}

impl<'a> BsmProblem<'a> {
  fn new(calibrator: &'a BsmCalibrator, start: BsmParams) -> Self {
    Self {
      calibrator,
      params: start,
      market: calibrator.quotes.iter().map(|quote| quote.price).collect(),
      trail: Trail::negated(),
    }
  }

  /// The stored iterate reflected, so even a raw start is priced inside the box.
  fn effective_params(&self) -> BsmParams {
    self.params.projected()
  }
}

impl LeastSquaresProblem for BsmProblem<'_> {
  /// Levenberg–Marquardt is unconstrained, so each iterate is stored reflected.
  fn set_params(&mut self, params: &Array1<f64>) {
    self.params = BsmParams::from(params.clone()).projected();
  }

  fn params(&self) -> Array1<f64> {
    self.effective_params().into()
  }

  fn residuals(&mut self) -> Option<Array1<f64>> {
    let BsmCalibrator {
      s,
      r,
      q,
      ref quotes,
      ref options,
      ..
    } = *self.calibrator;
    let params = self.effective_params();
    let model = BsmPricer::new(params.sigma, BsmCoc::Merton1973);
    let record = options.record_history;
    let mut prices = Vec::with_capacity(if record { quotes.len() } else { 0 });
    let mut residuals = Array1::zeros(quotes.len());

    for (idx, quote) in quotes.iter().enumerate() {
      let (call, put) = model.call_put(s, quote.strike, r, q, quote.tau);
      let price = match quote.option_type {
        OptionType::Call => call,
        OptionType::Put => put,
      };
      // Vega weighting approximates calibration in implied-volatility space.
      let vega = model
        .vega(s, quote.strike, r, q, quote.tau)
        .abs()
        .max_or_nan(1e-8);
      residuals[idx] = (price - quote.price) / vega;
      if record {
        prices.push(price);
      }
    }

    self
      .trail
      .stage(options, &self.market, &prices, &params, &residuals);
    Some(residuals)
  }

  fn jacobian(&self) -> Option<Array2<f64>> {
    // For vega-weighted residuals r = (C_model - C_mkt)/Vega,
    // dr/dsigma = 1 - r * (Vomma / Vega)
    let BsmCalibrator {
      s,
      r,
      q,
      ref quotes,
      ..
    } = *self.calibrator;
    let model = BsmPricer::new(self.effective_params().sigma, BsmCoc::Merton1973);
    let mut J = Array2::zeros((quotes.len(), 1));

    for (idx, quote) in quotes.iter().enumerate() {
      let (k, tau) = (quote.strike, quote.tau);
      let c_model_i = model.price_option(s, k, r, q, tau, quote.option_type);

      let vega = model.vega(s, k, r, q, tau).abs().max_or_nan(1e-8);
      let vomma = model.vomma(s, k, r, q, tau);
      let r_i = (c_model_i - quote.price) / vega;

      J[(idx, 0)] = 1.0 - r_i * (vomma / vega);
    }

    Some(J)
  }

  fn accept(&mut self) {
    self.trail.accept();
  }
}

#[cfg(test)]
mod tests;
