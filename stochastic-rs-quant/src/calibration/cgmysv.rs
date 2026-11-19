//! # CGMYSV Calibration
//!
//! $$
//! \hat\theta=\arg\min_\theta\sum_i\bigl(C_i^{\mathrm{model}}(\theta)-C_i^{\mathrm{mkt}}\bigr)^2
//! $$
//!
//! Calibrates the CGMYSV stochastic volatility model parameters
//! $(\alpha,\lambda_+,\lambda_-,\kappa,\eta,\zeta,\rho,v_0)$ to observed
//! option prices using Fourier pricing (Lewis) and Levenberg-Marquardt.
//!
//! Reference: Kim, Y. S. (2021), arXiv:2101.11001

use ndarray::Array1;
use ndarray::Array2;

use crate::OptionQuote;
use crate::OptionType;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationOptions;
use crate::calibration::clip_negative;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::Trail;
use crate::calibration::least_squares::minimize;
use crate::calibration::quote::assert_spot_and_rates;
use crate::calibration::quote::check_finite_start;
use crate::calibration::quote::validate_quotes;
use crate::pricing::cgmysv::CgmysvModel;
use crate::pricing::cgmysv::CgmysvParams;
use crate::pricing::fourier::LewisPricer;
use crate::traits::ToModel;

impl ToModel for Calibrated<CgmysvParams> {
  type Model = CgmysvModel;

  fn to_model(&self) -> CgmysvModel {
    CgmysvModel {
      params: self.params.clone(),
    }
  }
}

/// CGMYSV fit to vanilla quotes by Levenberg–Marquardt in logistic coordinates on the bounds.
#[derive(Clone, Debug)]
pub struct CgmysvCalibrator {
  s: f64,
  r: f64,
  q: f64,
  pub quotes: Vec<OptionQuote>,
  /// `None` starts from (0.5, 20, 5, 1, 0.05, 0.3, −1, 0.01); a start is clamped to the bounds.
  pub start: Option<CgmysvParams>,
  /// Levenberg–Marquardt knobs; `regularization` pulls `(α, λ₊, λ₋, κ, η, ζ, ρ, v0)`.
  pub options: CalibrationOptions,
}

impl CgmysvCalibrator {
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

  pub fn s(&self) -> f64 {
    self.s
  }

  pub fn r(&self) -> f64 {
    self.r
  }

  pub fn q(&self) -> f64 {
    self.q
  }

  /// Panics unless every value of `start` is finite.
  pub fn with_start(mut self, start: CgmysvParams) -> Self {
    check_start(&start).unwrap_or_else(|e| panic!("{e}"));
    self.start = Some(start);
    self
  }

  pub fn with_options(mut self, options: CalibrationOptions) -> Self {
    self.options = options;
    self
  }
}

const N_PARAMS: usize = 8;

/// `[alpha, lambda_plus, lambda_minus, kappa, eta, zeta, rho, v0]`, the optimiser's order.
fn to_cgmysv_params(p: &[f64]) -> CgmysvParams {
  CgmysvParams {
    alpha: p[0],
    lambda_plus: p[1],
    lambda_minus: p[2],
    kappa: p[3],
    eta: p[4],
    zeta: p[5],
    rho: p[6],
    v0: p[7],
  }
}

fn to_vec(p: &CgmysvParams) -> Vec<f64> {
  vec![
    p.alpha,
    p.lambda_plus,
    p.lambda_minus,
    p.kappa,
    p.eta,
    p.zeta,
    p.rho,
    p.v0,
  ]
}

fn check_start(p: &CgmysvParams) -> Result<(), QuantError> {
  let names = [
    "alpha",
    "lambda_plus",
    "lambda_minus",
    "kappa",
    "eta",
    "zeta",
    "rho",
    "v0",
  ];
  let fields = names.into_iter().zip(to_vec(p)).collect::<Vec<_>>();
  check_finite_start(&fields)
}

fn default_params() -> Vec<f64> {
  vec![0.5, 20.0, 5.0, 1.0, 0.05, 0.3, -1.0, 0.01]
}

fn param_bounds() -> [(f64, f64); N_PARAMS] {
  [
    (0.01, 1.999), // alpha
    (0.1, 100.0),  // lambda_plus
    (0.1, 100.0),  // lambda_minus
    (0.01, 10.0),  // kappa
    (0.001, 1.0),  // eta
    (0.01, 5.0),   // zeta
    (-5.0, 5.0),   // rho
    (0.0001, 0.5), // v0
  ]
}

fn project(p: &mut [f64]) {
  let bounds = param_bounds();
  for (v, (lo, hi)) in p.iter_mut().zip(bounds.iter()) {
    *v = v.clamp(*lo, *hi);
  }
}

/// Price a European call using the CGMYSV characteristic function via Lewis (2001).
fn fourier_call_price(params: &CgmysvParams, s: f64, k: f64, r: f64, q: f64, t: f64) -> f64 {
  let model = CgmysvModel {
    params: params.clone(),
  };
  LewisPricer::price_call(&model, s, k, r, q, t)
}

fn fourier_option_price(
  params: &CgmysvParams,
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  t: f64,
  option_type: OptionType,
) -> f64 {
  let call = fourier_call_price(params, s, k, r, q, t);
  match option_type {
    OptionType::Call => call,
    OptionType::Put => clip_negative(call - s * (-q * t).exp() + k * (-r * t).exp()),
  }
}

impl crate::traits::Calibrator for CgmysvCalibrator {
  type Params = CgmysvParams;
  type Output = Calibrated<CgmysvParams>;
  type Error = QuantError;

  /// A start replaces `self.start`; `InvalidInput` unless every value of the one used is finite,
  /// `Optimizer` when the fit cannot evaluate it, and the `CalibrationOptions` errors.
  fn calibrate(
    &self,
    initial: Option<CgmysvParams>,
  ) -> Result<Calibrated<CgmysvParams>, QuantError> {
    validate_quotes(&self.quotes)?;
    let start = initial.or_else(|| self.start.clone());
    if let Some(start) = &start {
      check_start(start)?;
    }
    let mut params = start.map_or_else(default_params, |start| to_vec(&start));
    project(&mut params);
    let lm = self.options.lm(true, N_PARAMS)?;
    let (fitted, report) = minimize(CgmysvProblem::new(self, params), lm);
    let model = fitted.compute_model_prices();
    report.calibrated(
      to_cgmysv_params(&fitted.params),
      &fitted.market,
      &model,
      self.options.loss_metrics,
      fitted.trail.into_entries(),
    )
  }
}

struct CgmysvProblem<'a> {
  calibrator: &'a CgmysvCalibrator,
  params: Vec<f64>,
  market: Vec<f64>,
  trail: Trail<CgmysvParams>,
}

impl<'a> CgmysvProblem<'a> {
  fn new(calibrator: &'a CgmysvCalibrator, params: Vec<f64>) -> Self {
    Self {
      calibrator,
      params,
      market: calibrator.quotes.iter().map(|quote| quote.price).collect(),
      trail: Trail::negated(),
    }
  }

  fn price(&self, params: &CgmysvParams, quote: &OptionQuote) -> f64 {
    let CgmysvCalibrator { s, r, q, .. } = *self.calibrator;
    fourier_option_price(params, s, quote.strike, r, q, quote.tau, quote.option_type)
  }

  fn compute_model_prices(&self) -> Vec<f64> {
    let p = to_cgmysv_params(&self.params);
    let quotes = self.calibrator.quotes.iter();
    quotes.map(|quote| self.price(&p, quote)).collect()
  }

  /// `∂θⱼ/∂xⱼ` of the logistic map from the optimiser's coordinates onto each parameter's bounds.
  fn logistic_factors(&self) -> impl Iterator<Item = f64> + '_ {
    let bounds = self.params.iter().zip(param_bounds());
    bounds.map(|(&value, (lower, upper))| (value - lower) * (upper - value) / (upper - lower))
  }
}

impl LeastSquaresProblem for CgmysvProblem<'_> {
  fn set_params(&mut self, params: &Array1<f64>) {
    self.params = params
      .iter()
      .zip(param_bounds())
      .map(|(&coordinate, (lower, upper))| lower + (upper - lower) / (1.0 + (-coordinate).exp()))
      .collect();
  }

  fn params(&self) -> Array1<f64> {
    // Smooth bounds avoid flat projected steps along weakly identified directions.
    // The interior margin also permits a finite start at either physical bound.
    Array1::from_iter(
      self
        .params
        .iter()
        .zip(param_bounds())
        .map(|(&value, (lower, upper))| {
          let margin = (upper - lower) * 1e-6;
          let value = value.clamp(lower + margin, upper - margin);
          ((value - lower) / (upper - value)).ln()
        }),
    )
  }

  fn residuals(&mut self) -> Option<Array1<f64>> {
    let model_prices = self.compute_model_prices();
    let rows = self.market.iter().zip(&model_prices);
    let residuals = Array1::from_iter(rows.map(|(market, model)| model - market));
    let options = &self.calibrator.options;
    let params = to_cgmysv_params(&self.params);
    self
      .trail
      .stage(options, &self.market, &model_prices, &params, &residuals);
    Some(residuals)
  }

  fn jacobian(&self) -> Option<Array2<f64>> {
    let n = self.calibrator.quotes.len();
    let mut jac = Array2::zeros((n, N_PARAMS));
    for j in 0..N_PARAMS {
      // The Lewis integral is adaptive to 1e-8, so a smaller step differences its noise.
      let h = 1e-5_f64.max(1e-3 * self.params[j].abs());
      let mut p_up = self.params.clone();
      let mut p_dn = self.params.clone();
      p_up[j] += h;
      p_dn[j] -= h;
      project(&mut p_up);
      project(&mut p_dn);

      let params_up = to_cgmysv_params(&p_up);
      let params_dn = to_cgmysv_params(&p_dn);
      let denom = p_up[j] - p_dn[j];
      if denom.abs() < 1e-15 {
        continue;
      }

      for (i, quote) in self.calibrator.quotes.iter().enumerate() {
        let price_up = self.price(&params_up, quote);
        let price_dn = self.price(&params_dn, quote);
        jac[(i, j)] = (price_up - price_dn) / denom;
      }
    }

    for (j, derivative) in self.logistic_factors().enumerate() {
      jac.column_mut(j).mapv_inplace(|v| v * derivative);
    }
    Some(jac)
  }

  fn accept(&mut self) {
    self.trail.accept();
  }

  fn natural_params(&self) -> Vec<f64> {
    self.params.clone()
  }

  fn natural_jacobian(&self) -> Array2<f64> {
    Array2::from_diag(&Array1::from_iter(self.logistic_factors()))
  }
}

#[cfg(test)]
mod tests;
