//! SABR fit to vanilla prices by Levenberg–Marquardt. A non-positive `s` panics in `new`; a
//! non-positive strike is `Err(InvalidQuote)`: unlike a caplet's rate, it has no shifted reading.
//!
//! $$
//! dF_t=\alpha_t F_t^\beta dW_t^1,\quad d\alpha_t=\nu\alpha_t dW_t^2,\ d\langle W^1,W^2\rangle_t=\rho dt
//! $$
//!
//! Hagan, Kumar, Lesniewski & Woodward (2002), "Managing Smile Risk", Wilmott Magazine, 84–108.

use ndarray::Array1;
use ndarray::Array2;

use crate::CalibrationLossScore;
use crate::OptionQuote;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationHistory;
use crate::calibration::CalibrationOptions;
use crate::calibration::Regularization;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::minimize;
use crate::calibration::quote::assert_spot_and_rates;
use crate::calibration::quote::validate_quotes;
use crate::pricing::sabr::SabrParams;
use crate::pricing::sabr::SabrPricer;
use crate::traits::ModelPricer;
use crate::traits::ToModel;

const RHO_BOUND: f64 = 0.9999;
const ALPHA_MIN: f64 = 1e-6;
const NU_MIN: f64 = 1e-6;
const DEFAULT_START: SabrParams = SabrParams {
  alpha: 0.2,
  beta: 1.0,
  rho: 0.0,
  nu: 0.8,
};

impl<D> ToModel for Calibrated<SabrParams, D> {
  type Model = SabrPricer;

  fn to_model(&self) -> SabrPricer {
    SabrPricer::new(self.params)
  }
}

impl SabrParams {
  pub fn project_in_place(&mut self) {
    self.alpha = self.alpha.abs().max(ALPHA_MIN);
    self.beta = self.beta.clamp(0.0, 1.0);
    self.nu = self.nu.abs().max(NU_MIN);
    self.rho = self.rho.clamp(-RHO_BOUND, RHO_BOUND);
  }

  pub fn projected(mut self) -> Self {
    self.project_in_place();
    self
  }
}

/// Lossless round-trip [α, β, ρ, ν]. Use `SabrParams::as_lm_vec` for the
/// 3-vec [α, ν, ρ] form the Levenberg-Marquardt solver operates on.
impl From<SabrParams> for Array1<f64> {
  fn from(p: SabrParams) -> Self {
    Array1::from_vec(vec![p.alpha, p.beta, p.rho, p.nu])
  }
}

/// Lossless round-trip from a 4-vec [α, β, ρ, ν]. Panics on a vector with
/// fewer than 4 elements (e.g. an LM 3-vec).
impl From<Array1<f64>> for SabrParams {
  fn from(v: Array1<f64>) -> Self {
    assert_eq!(
      v.len(),
      4,
      "SabrParams::from(Array1) expects 4 elements [alpha, beta, rho, nu], got {}",
      v.len()
    );
    SabrParams {
      alpha: v[0],
      beta: v[1],
      rho: v[2],
      nu: v[3],
    }
  }
}

impl SabrParams {
  /// 3-vec [α, ν, ρ] used by the LM optimiser. β is excluded because the
  /// calibrator does not optimise it — β is a user-set CEV exponent. Use the
  /// 4-vec [`From`] / [`Into`] for lossless round-trip.
  pub(crate) fn as_lm_vec(self) -> Array1<f64> {
    Array1::from_vec(vec![self.alpha, self.nu, self.rho])
  }
}

/// Levenberg–Marquardt fit of `(α, ν, ρ)` to vanilla quotes at the start's fixed `β`.
#[derive(Clone, Debug)]
pub struct SabrCalibrator {
  pub s: f64,
  pub r: f64,
  pub q: f64,
  pub quotes: Vec<OptionQuote>,
  /// `None` starts from `α = 0.2, β = 1, ρ = 0, ν = 0.8`.
  pub start: Option<SabrParams>,
  /// Levenberg–Marquardt knobs; `regularization` pulls Hagan's `(α, ρ, ν)`.
  pub options: CalibrationOptions,
}

impl SabrCalibrator {
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

  /// Panics unless `start` passes [`SabrParams::new`]; its `β` stays fixed through the fit.
  pub fn with_start(mut self, start: SabrParams) -> Self {
    self.start = Some(SabrParams::new(
      start.alpha,
      start.beta,
      start.rho,
      start.nu,
    ));
    self
  }

  pub fn with_options(mut self, options: CalibrationOptions) -> Self {
    self.options = options;
    self
  }
}

impl crate::traits::Calibrator for SabrCalibrator {
  type Params = SabrParams;
  type Output = Calibrated<SabrParams>;
  type Error = QuantError;

  /// A start replaces `self.start`; `InvalidInput` unless the one used passes `SabrParams::new`.
  fn calibrate(&self, initial: Option<SabrParams>) -> Result<Calibrated<SabrParams>, QuantError> {
    validate_quotes(&self.quotes)?;
    let start = initial.or(self.start);
    if let Some(reason) = start.and_then(|start| start.violation()) {
      return Err(QuantError::invalid("start", reason));
    }
    let problem = SabrProblem {
      calibrator: self,
      market: self.quotes.iter().map(|quote| quote.price).collect(),
      params: start.unwrap_or(DEFAULT_START).projected(),
      regularization: self.options.regularization_for(3)?.map(optimiser_order),
      history: Vec::new(),
      pending: None,
    };
    let (problem, report) = minimize(problem, self.options.lm(false));
    let model = problem.model_prices(&problem.params);
    Ok(report.calibrated(
      problem.params,
      &problem.market,
      &model,
      self.options.loss_metrics,
      problem.history,
    ))
  }
}

/// Hagan's `(α, ρ, ν)` anchor and weights in the optimiser's `(α, ν, ρ)` order.
fn optimiser_order(regularization: Regularization) -> Regularization {
  let reorder = |v: &[f64]| vec![v[0], v[2], v[1]];
  Regularization::new(
    reorder(&regularization.anchor),
    reorder(&regularization.weights),
  )
}

/// One fit: the current point, the regularization in the optimiser's order and the trail.
struct SabrProblem<'a> {
  calibrator: &'a SabrCalibrator,
  market: Vec<f64>,
  params: SabrParams,
  regularization: Option<Regularization>,
  history: Vec<CalibrationHistory<SabrParams>>,
  pending: Option<CalibrationHistory<SabrParams>>,
}

impl SabrProblem<'_> {
  fn model_prices(&self, p: &SabrParams) -> Vec<f64> {
    let pricer = SabrPricer::new(*p);
    let SabrCalibrator { s, r, q, .. } = *self.calibrator;
    self
      .calibrator
      .quotes
      .iter()
      .map(|quote| pricer.price_option(s, quote.strike, r, q, quote.tau, quote.option_type))
      .collect()
  }

  fn residuals_for(&self, p: &SabrParams) -> Array1<f64> {
    let model = self.model_prices(p);
    Array1::from_iter(self.market.iter().zip(&model).map(|(m, q)| m - q))
  }

  fn numeric_jacobian(&self, p: &SabrParams) -> Array2<f64> {
    let base = p.as_lm_vec();
    let mut jacobian = Array2::zeros((self.market.len(), base.len()));
    for col in 0..base.len() {
      let x = base[col];
      let mut h = 1e-5_f64.max(1e-3 * x.abs());
      let mut p_plus = *p;
      let mut p_minus = *p;
      match col {
        0 => {
          p_plus.alpha = (x + h).abs().max(ALPHA_MIN);
          p_minus.alpha = (x - h).abs().max(ALPHA_MIN);
        }
        1 => {
          p_plus.nu = (x + h).abs().max(NU_MIN);
          p_minus.nu = (x - h).abs().max(NU_MIN);
        }
        2 => {
          let clamp = |y: f64| y.clamp(-RHO_BOUND, RHO_BOUND);
          p_plus.rho = clamp(x + h);
          p_minus.rho = clamp(x - h);
          if (p_plus.rho - p_minus.rho).abs() < 0.5 * h {
            h = 1e-4;
            p_plus.rho = clamp(x + h);
            p_minus.rho = clamp(x - h);
          }
        }
        _ => unreachable!(),
      }
      p_plus.project_in_place();
      p_minus.project_in_place();
      let diff = (self.residuals_for(&p_plus) - self.residuals_for(&p_minus)) / (2.0 * h);
      jacobian.column_mut(col).assign(&diff);
    }
    jacobian
  }
}

impl LeastSquaresProblem for SabrProblem<'_> {
  fn set_params(&mut self, x: &Array1<f64>) {
    self.params = SabrParams {
      alpha: x[0],
      beta: self.params.beta,
      rho: x[2],
      nu: x[1],
    }
    .projected();
  }

  fn params(&self) -> Array1<f64> {
    self.params.as_lm_vec()
  }

  fn residuals(&mut self) -> Option<Array1<f64>> {
    let model = self.model_prices(&self.params);
    let residuals = Array1::from_iter(self.market.iter().zip(&model).map(|(m, q)| m - q));
    let options = &self.calibrator.options;
    if options.record_history {
      let loss = CalibrationLossScore::compute_selected(&self.market, &model, options.loss_metrics);
      let entry = CalibrationHistory::new(self.history.len(), self.params, residuals.clone(), loss);
      self.pending = Some(entry);
    }
    let p = self.params;
    match &self.regularization {
      Some(reg) if reg.is_active() => {
        Some(reg.augment_residuals(residuals, &[p.alpha, p.nu, p.rho]))
      }
      _ => Some(residuals),
    }
  }

  fn jacobian(&self) -> Option<Array2<f64>> {
    let jacobian = self.numeric_jacobian(&self.params);
    match &self.regularization {
      Some(reg) if reg.is_active() => Some(reg.augment_jacobian(jacobian, reg.jacobian_rows())),
      _ => Some(jacobian),
    }
  }

  fn accept(&mut self) {
    self.history.extend(self.pending.take());
  }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod regularization_tests;
