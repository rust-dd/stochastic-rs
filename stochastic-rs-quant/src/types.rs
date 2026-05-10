//! # Shared types
//!
//! Cross-cutting enums and result containers used across the quant crate:
//! option taxonomy ([`OptionType`], [`OptionStyle`], [`Moneyness`]) and
//! calibration loss metrics ([`LossMetric`], [`CalibrationLossScore`]).
//!
//! These types are also re-exported at the crate root for back-compat with
//! v1 call-sites.

use std::collections::HashMap;
use std::fmt::Display;

use crate::loss;

/// Option type.
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum OptionType {
  #[default]
  Call,
  Put,
}

/// Option style.
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum OptionStyle {
  American,
  #[default]
  European,
}

/// Moneyness.
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Moneyness {
  #[default]
  DeepInTheMoney,
  InTheMoney,
  AtTheMoney,
  OutOfTheMoney,
  DeepOutOfTheMoney,
}

impl Display for Moneyness {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Moneyness::DeepInTheMoney => write!(f, "Deep in the money"),
      Moneyness::InTheMoney => write!(f, "In the money"),
      Moneyness::AtTheMoney => write!(f, "At the money"),
      Moneyness::OutOfTheMoney => write!(f, "Out of the money"),
      Moneyness::DeepOutOfTheMoney => write!(f, "Deep out of the money"),
    }
  }
}

/// Individual loss metric selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LossMetric {
  Mae,
  Mse,
  Rmse,
  Mpe,
  Mape,
  Mspe,
  Rmspe,
  Mre,
  Mrpe,
}

impl LossMetric {
  pub const ALL: [Self; 9] = [
    Self::Mae,
    Self::Mse,
    Self::Rmse,
    Self::Mpe,
    Self::Mape,
    Self::Mspe,
    Self::Rmspe,
    Self::Mre,
    Self::Mrpe,
  ];

  fn compute(self, market: &[f64], model: &[f64]) -> f64 {
    match self {
      Self::Mae => loss::mae(market, model),
      Self::Mse => loss::mse(market, model),
      Self::Rmse => loss::rmse(market, model),
      Self::Mpe => loss::mpe(market, model),
      Self::Mape => loss::mape(market, model),
      Self::Mspe => loss::mspe(market, model),
      Self::Rmspe => loss::rmspe(market, model),
      Self::Mre => loss::mre(market, model),
      Self::Mrpe => loss::mrpe(market, model),
    }
  }
}

/// Holds calibration loss metrics as a `HashMap<LossMetric, f64>`.
#[derive(Default, Debug, Clone)]
pub struct CalibrationLossScore {
  pub scores: HashMap<LossMetric, f64>,
}

impl CalibrationLossScore {
  /// Compute all loss metrics.
  pub fn compute(market: &[f64], model: &[f64]) -> Self {
    Self::compute_selected(market, model, &LossMetric::ALL)
  }

  /// Compute only the selected metrics.
  pub fn compute_selected(market: &[f64], model: &[f64], metrics: &[LossMetric]) -> Self {
    let scores = metrics
      .iter()
      .map(|&m| (m, m.compute(market, model)))
      .collect();
    Self { scores }
  }

  /// Get a metric value. Returns `f64::NAN` when the metric was not
  /// computed (e.g. a calibrator that only fills `LossMetric::Rmse` on a
  /// `LossMetric::Mae` lookup). Use [`Self::try_get`] for an explicit
  /// `Option<f64>`.
  pub fn get(&self, metric: LossMetric) -> f64 {
    self.scores.get(&metric).copied().unwrap_or(f64::NAN)
  }

  /// Get a metric value as `Option<f64>` — `None` when the metric was
  /// not computed by the calibrator. Prefer this when downstream code
  /// needs to branch on presence (rather than treating absence as zero
  /// or NaN).
  pub fn try_get(&self, metric: LossMetric) -> Option<f64> {
    self.scores.get(&metric).copied()
  }
}
