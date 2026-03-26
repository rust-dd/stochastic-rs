//! # Quant
//!
//! $$
//! dX_t=a(t,X_t)dt+b(t,X_t)dW_t
//! $$
//!
use std::collections::HashMap;
use std::fmt::Display;

pub mod bonds;
pub mod calendar;
pub mod calibration;
pub mod curves;
pub mod fourier_malliavin;
pub mod fx;
pub mod loss;
pub mod order_book;
pub mod portfolio;
pub mod pricing;
pub mod strategies;
pub mod vol_surface;
pub use portfolio::momentum;
#[cfg(feature = "yahoo")]
pub mod yahoo;

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

  /// Get a metric value. Returns 0.0 if not computed.
  pub fn get(&self, metric: LossMetric) -> f64 {
    self.scores.get(&metric).copied().unwrap_or(0.0)
  }
}
