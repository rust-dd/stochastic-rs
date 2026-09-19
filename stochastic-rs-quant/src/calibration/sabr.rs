//! # Sabr
//!
//! Price-based Sabr calibrator using Levenberg-Marquardt.
//!
//! **Reference:** P. S. Hagan, D. Kumar, A. S. Lesniewski, D. E. Woodward,
//! *Managing Smile Risk*, Wilmott Magazine, pp. 84–108, 2002.
//!
//! $$
//! dF_t=\alpha_t F_t^\beta dW_t^1,\quad d\alpha_t=\nu\alpha_t dW_t^2,\ d\langle W^1,W^2\rangle_t=\rho dt
//! $$
//!
//! ## Non-positive spot or strike
//!
//! [`hagan_implied_vol`](crate::pricing::sabr::hagan_implied_vol) requires a
//! strictly positive forward and strike, and `s`/`k` here feed both
//! directly. [`SabrCapletCalibrator`](crate::calibration::sabr_caplet::SabrCapletCalibrator)
//! handles the analogous problem with a displacement shift because a
//! negative interest-rate forward is a genuine market state (EUR, JPY and
//! CHF caplet forwards were routinely negative from roughly 2015 to 2022).
//! That reasoning does not transfer here: `s`/`k` are an equity or FX spot
//! and strike, so a non-positive value is not a market state, it is invalid
//! input. [`Calibrator::calibrate`](crate::traits::Calibrator::calibrate)
//! therefore rejects it outright — `Err` naming the offending quote —
//! rather than reinterpreting it in shifted coordinates. Do not
//! "harmonise" this calibrator with the caplet one's shift; the two fixes
//! address the same symptom for different, non-interchangeable reasons.
//!
use std::cell::RefCell;
use std::rc::Rc;

use ndarray::Array1;
use ndarray::Array2;

use crate::CalibrationLossScore;
use crate::LossMetric;
use crate::OptionType;
use crate::calibration::CalibrationHistory;
use crate::calibration::Regularization;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::LmOptions;
use crate::calibration::least_squares::minimize;
use crate::pricing::sabr::SabrPricer;

const RHO_BOUND: f64 = 0.9999;
const ALPHA_MIN: f64 = 1e-6;
const NU_MIN: f64 = 1e-6;

/// Calibration result for the Sabr model.
#[derive(Clone, Debug)]
pub struct SabrCalibrationResult {
  pub alpha: f64,
  pub beta: f64,
  pub nu: f64,
  pub rho: f64,
  /// Calibration loss metrics.
  pub loss: CalibrationLossScore,
  /// Whether the optimiser converged.
  pub converged: bool,
}

impl crate::traits::ToModel for SabrCalibrationResult {
  type Model = crate::pricing::sabr::SabrPricer;
  fn to_model(&self, _r: f64, _q: f64) -> Self::Model {
    SabrCalibrationResult::to_model(self)
  }
}

impl SabrCalibrationResult {
  /// Convert to a [`SabrPricer`] for
  /// pricing / vol surface generation.
  pub fn to_model(&self) -> crate::pricing::sabr::SabrPricer {
    crate::pricing::sabr::SabrPricer {
      alpha: self.alpha,
      beta: self.beta,
      nu: self.nu,
      rho: self.rho,
    }
  }
}

impl crate::traits::CalibrationResult for SabrCalibrationResult {
  type Params = SabrParams;
  fn rmse(&self) -> f64 {
    self.loss.get(LossMetric::Rmse)
  }

  fn converged(&self) -> bool {
    self.converged
  }

  fn params(&self) -> Self::Params {
    SabrParams {
      alpha: self.alpha,
      beta: self.beta,
      nu: self.nu,
      rho: self.rho,
    }
  }

  fn loss_score(&self) -> Option<&CalibrationLossScore> {
    Some(&self.loss)
  }
}

impl crate::traits::Calibrator for SabrCalibrator {
  type InitialGuess = SabrParams;
  type Params = SabrParams;
  type Output = SabrCalibrationResult;
  type Error = anyhow::Error;

  fn calibrate(&self, initial: Option<Self::InitialGuess>) -> Result<Self::Output, Self::Error> {
    let mut this = self.clone();
    if let Some(p) = initial {
      this.params = Some(p.projected());
    }
    this.validate_market_data()?;
    Ok(this.solve())
  }
}

#[derive(Clone, Copy, Debug)]
pub struct SabrParams {
  /// Model shape/loading parameter.
  pub alpha: f64,
  /// Cev exponent (0 = normal, 1 = lognormal).
  pub beta: f64,
  /// Volatility-of-volatility parameter.
  pub nu: f64,
  /// Correlation parameter.
  pub rho: f64,
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

/// Lossless round-trip [α, β, ν, ρ]. Use `SabrParams::as_lm_vec` for the
/// 3-vec [α, ν, ρ] form the Levenberg-Marquardt solver operates on.
impl From<SabrParams> for Array1<f64> {
  fn from(p: SabrParams) -> Self {
    Array1::from_vec(vec![p.alpha, p.beta, p.nu, p.rho])
  }
}

/// Lossless round-trip from a 4-vec [α, β, ν, ρ]. Panics on a vector with
/// fewer than 4 elements (e.g. an LM 3-vec).
impl From<Array1<f64>> for SabrParams {
  fn from(v: Array1<f64>) -> Self {
    assert_eq!(
      v.len(),
      4,
      "SabrParams::from(Array1) expects 4 elements [alpha, beta, nu, rho], got {}",
      v.len()
    );
    SabrParams {
      alpha: v[0],
      beta: v[1],
      nu: v[2],
      rho: v[3],
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

#[derive(Clone)]
pub struct SabrCalibrator {
  /// Model parameter set (input or calibrated output).
  pub params: Option<SabrParams>,
  /// Observed market option prices used for calibration.
  pub c_market: Array1<f64>,
  /// Underlying spot/forward level.
  pub s: Array1<f64>,
  /// Strike level.
  pub k: Array1<f64>,
  /// Risk-free rate used for discounting.
  pub r: f64,
  /// Dividend yield / convenience yield.
  pub q: Option<f64>,
  /// Time to maturity in years.
  pub tau: f64,
  /// Option direction (call/put).
  pub option_type: OptionType,
  /// If true, stores optimization parameter history.
  pub record_history: bool,
  /// Which loss metrics to compute when recording history.
  pub loss_metrics: &'static [LossMetric],
  /// Optional Tikhonov pull of `(α, ν, ρ)` toward an anchor; `None` keeps
  /// the unregularised path.
  pub regularization: Option<Regularization>,
  calibration_history: Rc<RefCell<Vec<CalibrationHistory<SabrParams>>>>,
}

impl SabrCalibrator {
  pub fn new(
    params: Option<SabrParams>,
    c_market: Array1<f64>,
    s: Array1<f64>,
    k: Array1<f64>,
    r: f64,
    q: Option<f64>,
    tau: f64,
    option_type: OptionType,
    record_history: bool,
  ) -> Self {
    assert!(
      s.len() == k.len() && k.len() == c_market.len(),
      "s, k and c_market must have equal length, got s.len() = {}, k.len() = {}, c_market.len() = {}",
      s.len(),
      k.len(),
      c_market.len()
    );
    Self {
      params,
      c_market,
      s,
      k,
      r,
      q,
      tau,
      option_type,
      record_history,
      loss_metrics: &LossMetric::ALL,
      regularization: None,
      calibration_history: Rc::new(RefCell::new(Vec::new())),
    }
  }

  /// Adds a Tikhonov pull toward `regularization.anchor` in the natural
  /// order `(α, ν, ρ)`; β stays fixed.
  pub fn with_regularization(mut self, regularization: Regularization) -> Self {
    assert_eq!(
      regularization.dimension(),
      3,
      "SABR regularisation needs three anchors"
    );
    self.regularization = Some(regularization);
    self
  }
}

impl SabrCalibrator {
  /// Checks that every spot/forward level and strike is finite and
  /// strictly positive, as the underlying Hagan expansion requires. Called
  /// from [`Calibrator::calibrate`](crate::traits::Calibrator::calibrate)
  /// so bad market data surfaces as `Err` naming the offending quote
  /// instead of panicking inside the Levenberg-Marquardt cost callback —
  /// see the module documentation for why this calibrator rejects rather
  /// than shifts. Checked as `!x.is_finite() || x <= 0.0` — matching
  /// [`RBergomiCalibrator`](crate::calibration::rbergomi::RBergomiCalibrator)'s
  /// own `is_finite() && x > 0.0` precondition on its scalar inputs — so
  /// `NaN` is rejected too: a plain `x <= 0.0` silently lets `NaN` through,
  /// since every comparison against `NaN` is false.
  fn validate_market_data(&self) -> Result<(), anyhow::Error> {
    for (i, &s) in self.s.iter().enumerate() {
      if !s.is_finite() || s <= 0.0 {
        anyhow::bail!("SabrCalibrator: s[{i}] must be strictly positive (got {s})");
      }
    }
    for (i, &k) in self.k.iter().enumerate() {
      if !k.is_finite() || k <= 0.0 {
        anyhow::bail!("SabrCalibrator: k[{i}] must be strictly positive (got {k})");
      }
    }
    Ok(())
  }

  fn solve(&self) -> SabrCalibrationResult {
    let mut problem = self.clone();
    problem.ensure_initial_guess();

    let (result, report) = minimize(
      problem,
      LmOptions {
        pivoted_qr: false,
        ..LmOptions::default()
      },
    );
    let converged = report.converged;
    let p = result.effective_params();
    let c_model = result.compute_model_prices_for(&p);
    let loss = CalibrationLossScore::compute_selected(
      result.c_market.as_standard_layout().as_slice().unwrap(),
      c_model.as_slice().unwrap(),
      result.loss_metrics,
    );

    SabrCalibrationResult {
      alpha: p.alpha,
      beta: p.beta,
      nu: p.nu,
      rho: p.rho,
      loss,
      converged,
    }
  }

  pub fn set_initial_guess(&mut self, params: SabrParams) {
    self.params = Some(params.projected());
  }

  pub fn set_record_history(&mut self, record: bool) {
    self.record_history = record;
  }

  pub fn history(&self) -> Vec<CalibrationHistory<SabrParams>> {
    self.calibration_history.borrow().clone()
  }

  fn ensure_initial_guess(&mut self) {
    if self.params.is_none() {
      self.params = Some(
        SabrParams {
          alpha: 0.2,
          beta: 1.0,
          nu: 0.8,
          rho: 0.0,
        }
        .projected(),
      );
    }
  }

  fn effective_params(&self) -> SabrParams {
    if let Some(p) = &self.params {
      return (*p).projected();
    }
    SabrParams {
      alpha: 0.2,
      beta: 1.0,
      nu: 0.8,
      rho: 0.0,
    }
    .projected()
  }

  fn compute_model_prices_for(&self, p: &SabrParams) -> Array1<f64> {
    let mut c_model = Array1::zeros(self.c_market.len());
    for i in 0..self.c_market.len() {
      let pr = SabrPricer::new(p.alpha, p.beta, p.nu, p.rho);
      let (call, put) = pr.call_put(
        self.s[i],
        self.k[i],
        self.r,
        self.q.unwrap_or(0.0),
        self.tau,
      );
      c_model[i] = match self.option_type {
        OptionType::Call => call,
        OptionType::Put => put,
      };
    }
    c_model
  }

  fn residuals_for(&self, p: &SabrParams) -> Array1<f64> {
    self.c_market.clone() - self.compute_model_prices_for(p)
  }

  fn numeric_jacobian(&self, p: &SabrParams) -> Array2<f64> {
    let n = self.c_market.len();
    let m = 3usize; // alpha, nu, rho
    let base: Array1<f64> = p.as_lm_vec();
    let mut J = Array2::zeros((n, m));
    for col in 0..m {
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
      let r_plus = self.residuals_for(&p_plus);
      let r_minus = self.residuals_for(&p_minus);
      let diff = (r_plus - r_minus) / (2.0 * h);
      for row in 0..n {
        J[(row, col)] = diff[row];
      }
    }
    J
  }
}

impl LeastSquaresProblem for SabrCalibrator {
  fn set_params(&mut self, params: &Array1<f64>) {
    let beta = self.effective_params().beta;
    let mut p = SabrParams {
      alpha: params[0],
      beta,
      nu: params[1],
      rho: params[2],
    };
    p.project_in_place();
    self.params = Some(p);
  }

  fn params(&self) -> Array1<f64> {
    self.effective_params().as_lm_vec()
  }

  fn residuals(&self) -> Option<Array1<f64>> {
    let p = self.effective_params();
    let c_model = self.compute_model_prices_for(&p);
    if self.record_history {
      self
        .calibration_history
        .borrow_mut()
        .push(CalibrationHistory {
          residuals: self.c_market.clone() - c_model.clone(),
          call_put: self
            .c_market
            .iter()
            .enumerate()
            .map(|(i, _)| {
              let pr = SabrPricer::new(p.alpha, p.beta, p.nu, p.rho);
              pr.call_put(
                self.s[i],
                self.k[i],
                self.r,
                self.q.unwrap_or(0.0),
                self.tau,
              )
            })
            .collect::<Vec<(f64, f64)>>()
            .into(),
          params: p,
          loss_scores: CalibrationLossScore::compute_selected(
            self.c_market.as_standard_layout().as_slice().unwrap(),
            c_model.as_slice().unwrap(),
            self.loss_metrics,
          ),
        });
    }
    let residuals = self.c_market.clone() - c_model;
    match &self.regularization {
      Some(reg) if reg.is_active() => {
        Some(reg.augment_residuals(residuals, &[p.alpha, p.nu, p.rho]))
      }
      _ => Some(residuals),
    }
  }

  fn jacobian(&self) -> Option<Array2<f64>> {
    let jacobian = self.numeric_jacobian(&self.effective_params());
    match &self.regularization {
      Some(reg) if reg.is_active() => Some(reg.augment_jacobian(jacobian, reg.jacobian_rows())),
      _ => Some(jacobian),
    }
  }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod regularization_tests;
