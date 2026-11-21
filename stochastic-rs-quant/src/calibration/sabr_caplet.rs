//! Per-expiry SABR smile fit for caplets and swaptions: Nelder–Mead on `(α, ν, ρ)` at a fixed `β`
//! against Black-76 vols, the forward and strikes displaced by the start's shift.
//!
//! $$
//! (\hat\alpha,\hat\nu,\hat\rho)=\arg\min_{\alpha,\nu,\rho}
//!   \sum_i w_i\bigl(\sigma_{\mathrm{Hagan}}(K_i,F,\tau;\alpha,\beta,\nu,\rho)
//!   -\sigma_i^{mkt}\bigr)^2
//! $$
//!
//! Reference: Hagan et al. (2002), Wilmott Mag., 84–108
//! Reference: Obłój (2008), Wilmott Mag.

use std::convert::Infallible;

use basin::CostFunction;
use stochastic_rs_distributions::RealExt;

use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationOptions;
use crate::calibration::Regularization;
use crate::calibration::nelder_mead_fit;
use crate::calibration::start_vertex;
use crate::instruments::option::ShiftedSabrVolatility;
use crate::pricing::sabr::SabrParams;
use crate::pricing::sabr::SabrPricer;
use crate::pricing::sabr::hagan_implied_vol;
use crate::traits::ToModel;

/// The caplet smile's Hagan parameters and the displacement they were calibrated under.
#[derive(Debug, Clone, Copy)]
pub struct SabrCapletParams {
  pub sabr: SabrParams,
  /// Displacement of forward and strikes; `sabr` is in shifted coordinates when it is nonzero.
  pub shift: f64,
}

impl SabrCapletParams {
  /// The first reason these cannot start a fit: `SabrParams::new`'s checks, then a finite shift.
  fn violation(&self) -> Option<String> {
    let shift = self.shift;
    self.sabr.violation().or_else(|| {
      (!shift.is_finite())
        .then(|| format!("shift must satisfy `shift.is_finite()`, got shift = {shift:?}"))
    })
  }
}

impl ToModel for Calibrated<SabrCapletParams> {
  type Model = SabrPricer;

  /// Unshifted: with a nonzero shift, shift `s` and `k` yourself (non-positive ones price NaN) or
  /// use [`Self::to_shifted_volatility`].
  fn to_model(&self) -> SabrPricer {
    SabrPricer::new(self.params.sabr)
  }
}

impl Calibrated<SabrCapletParams> {
  /// The fit as a [`ShiftedSabrVolatility`]: Black prices it on the shifted forward and strike.
  pub fn to_shifted_volatility(&self) -> ShiftedSabrVolatility<f64> {
    let SabrCapletParams { sabr, shift } = self.params;
    ShiftedSabrVolatility::new(sabr.alpha, sabr.beta, sabr.rho, sabr.nu, shift)
  }
}

/// Nelder–Mead fit of one expiry's Black-76 vol smile.
#[derive(Debug, Clone)]
pub struct SabrCapletCalibrator {
  pub forward: f64,
  /// Expiry in years.
  pub tau: f64,
  pub strikes: Vec<f64>,
  /// Black-76 implied vols, one per strike.
  pub market_vols: Vec<f64>,
  /// Weights on the squared vol errors, one per strike; `None` weighs each strike 1.
  pub weights: Option<Vec<f64>>,
  /// Seeds `(α, ν, ρ)` and fixes `β` and the shift.
  pub start: SabrCapletParams,
  /// Nelder–Mead: `max_iterations` (default 600), `tolerance` its spread; pull `(α, ρ, ν)`, and
  /// `patience` is `Unsupported`.
  pub options: CalibrationOptions,
}

impl SabrCapletCalibrator {
  /// Panics unless `market_vols` has one vol per strike and `start.sabr` passes
  /// [`SabrParams::new`] with a finite `start.shift`.
  pub fn new(
    forward: f64,
    tau: f64,
    strikes: Vec<f64>,
    market_vols: Vec<f64>,
    start: SabrCapletParams,
  ) -> Self {
    assert!(
      market_vols.len() == strikes.len(),
      "market_vols must satisfy `market_vols.len() == strikes.len()`, got market_vols.len() = {} \
       for {} strikes",
      market_vols.len(),
      strikes.len()
    );
    if let Some(reason) = start.violation() {
      panic!("{reason}");
    }
    Self {
      forward,
      tau,
      strikes,
      market_vols,
      weights: None,
      start,
      options: CalibrationOptions::default(),
    }
  }

  /// Panics unless there is one weight per strike, each finite and non-negative.
  pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
    assert!(
      weights.len() == self.strikes.len(),
      "weights must satisfy `weights.len() == strikes.len()`, got weights.len() = {} for {} strikes",
      weights.len(),
      self.strikes.len()
    );
    if let Some(w) = weights.iter().find(|w| !(w.is_finite() && **w >= 0.0)) {
      panic!("weights must satisfy `w.is_finite() && w >= 0.0` for every w, got w = {w:?}");
    }
    self.weights = Some(weights);
    self
  }

  pub fn with_options(mut self, options: CalibrationOptions) -> Self {
    self.options = options;
    self
  }

  /// `Err` for the first datum the fit cannot use under `shift`.
  fn validate(&self, shift: f64) -> Result<(), QuantError> {
    if self.strikes.is_empty() {
      return Err(QuantError::InsufficientData {
        what: "caplet vols",
        needed: 1,
        got: 0,
      });
    }
    let (tau, forward) = (self.tau, self.forward);
    if !(tau.is_finite() && tau > 0.0) {
      let reason = format!("tau must satisfy `tau.is_finite() && tau > 0.0`, got tau = {tau:?}");
      return Err(QuantError::invalid("tau", reason));
    }
    if !forward.is_finite() {
      let reason = format!("forward must satisfy `forward.is_finite()`, got forward = {forward:?}");
      return Err(QuantError::invalid("forward", reason));
    }
    if forward + shift <= 0.0 {
      let reason = format!("shift {shift} leaves forward {forward} non-positive");
      return Err(QuantError::invalid("forward", reason));
    }
    for (index, (&k, &vol)) in self.strikes.iter().zip(&self.market_vols).enumerate() {
      let reason = if !k.is_finite() {
        format!("strike must satisfy `strike.is_finite()`, got strike = {k:?}")
      } else if k + shift <= 0.0 {
        format!("shift {shift} leaves strike {k} non-positive")
      } else if !(vol.is_finite() && vol >= 0.0) {
        format!("vol must satisfy `vol.is_finite() && vol >= 0.0`, got vol = {vol:?}")
      } else {
        continue;
      };
      return Err(QuantError::InvalidQuote { index, reason });
    }
    let Some(weights) = &self.weights else {
      return Ok(());
    };
    if weights.len() != self.strikes.len() {
      return Err(QuantError::DimensionMismatch {
        what: "weights, one per strike",
        expected: self.strikes.len(),
        got: weights.len(),
      });
    }
    match weights
      .iter()
      .enumerate()
      .find(|(_, w)| !(w.is_finite() && **w >= 0.0))
    {
      Some((index, w)) => Err(QuantError::invalid(
        "weights",
        format!("weight {index} must satisfy `w.is_finite() && w >= 0.0`, got w = {w:?}"),
      )),
      None => Ok(()),
    }
  }
}

impl crate::traits::Calibrator for SabrCapletCalibrator {
  type Params = SabrCapletParams;
  type Output = Calibrated<SabrCapletParams>;
  type Error = QuantError;

  /// A start replaces `self.start` whole, `β` and the shift included. Errors: `InvalidInput`,
  /// `InvalidQuote`, `InsufficientData`, `DimensionMismatch` and the `CalibrationOptions` ones.
  fn calibrate(&self, initial: Option<SabrCapletParams>) -> Result<Self::Output, QuantError> {
    let start = initial.unwrap_or(self.start);
    if let Some(reason) = start.violation() {
      return Err(QuantError::invalid("start", reason));
    }
    self.validate(start.shift)?;
    let cost = SabrCapletCost {
      forward: self.forward,
      shift: start.shift,
      tau: self.tau,
      beta: start.sabr.beta,
      strikes: self.strikes.clone(),
      market_vols: self.market_vols.clone(),
      weights: self
        .weights
        .clone()
        .unwrap_or_else(|| vec![1.0; self.strikes.len()]),
      regularization: self.options.regularization_for(3)?,
    };
    let SabrParams {
      alpha: a0,
      nu: nu0,
      rho: rho0,
      ..
    } = start.sabr;
    let simplex = vec![
      vec![a0, nu0, rho0],
      vec![start_vertex(a0, 1.3), nu0, rho0],
      vec![a0, start_vertex(nu0, 1.5), rho0],
      vec![a0, nu0, (rho0 + 0.2).clamp(-0.99, 0.99)],
    ];
    nelder_mead_fit(cost.clone(), simplex, &self.options, 600, |x| {
      let sabr = cost.params(x);
      let fitted = SabrCapletParams {
        sabr,
        shift: start.shift,
      };
      (fitted, self.market_vols.clone(), cost.model_vols(&sabr))
    })
  }
}

#[derive(Clone)]
struct SabrCapletCost {
  forward: f64,
  shift: f64,
  tau: f64,
  beta: f64,
  strikes: Vec<f64>,
  market_vols: Vec<f64>,
  weights: Vec<f64>,
  /// Hagan's `(α, ρ, ν)` order.
  regularization: Option<Regularization>,
}

impl SabrCapletCost {
  fn params(&self, x: &[f64]) -> SabrParams {
    SabrParams {
      alpha: x[0].abs().max_or_nan(1e-8),
      beta: self.beta,
      rho: x[2].clamp(-0.9999, 0.9999),
      nu: x[1].abs().max_or_nan(1e-8),
    }
  }

  /// NaN for parameters `SabrParams::new` refuses; `hagan_implied_vol` panics on a bad `α` or `ρ`.
  fn model_vols(&self, params: &SabrParams) -> Vec<f64> {
    if params.violation().is_some() {
      return vec![f64::NAN; self.strikes.len()];
    }
    let forward = self.forward + self.shift;
    self
      .strikes
      .iter()
      .map(|&k| hagan_implied_vol(k + self.shift, forward, self.tau, params))
      .collect()
  }
}

impl CostFunction for SabrCapletCost {
  type Param = Vec<f64>;
  type Output = f64;
  type Error = Infallible;

  fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
    let params = self.params(x);
    let mut sse = 0.0;
    for ((v_model, &v_mkt), &w) in self
      .model_vols(&params)
      .into_iter()
      .zip(&self.market_vols)
      .zip(&self.weights)
    {
      let diff = v_model - v_mkt;
      sse += w * diff * diff;
    }
    let penalty = self.regularization.as_ref().map_or(0.0, |reg| {
      reg.penalty(&[params.alpha, params.rho, params.nu])
    });
    Ok(sse + penalty)
  }
}

#[cfg(test)]
mod tests;
