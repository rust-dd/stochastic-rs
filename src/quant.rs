use std::fmt::Display;

pub mod bonds;
pub mod calibration;
pub mod pricing;
pub mod strategies;
pub mod r#trait;
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

/// Holds various calibration loss metrics in one place.
#[derive(Default, Debug, Clone, Copy)]
pub struct CalibrationLossScore {
  /// Mean Absolute Error
  pub mae: f64,
  /// Mean Squared Error
  pub mse: f64,
  /// Root Mean Squared Error
  pub rmse: f64,
  /// Mean Percentage Error (in %)
  pub mpe: f64,
  /// Mean Absolute Percentage Error (in %)
  pub mape: f64,
  /// Mean Squared Percentage Error (in %)
  pub mspe: f64,
  /// Root Mean Squared Percentage Error (in %)
  pub rmspe: f64,
  /// Mean Relative Error (no %)
  pub mre: f64,
  /// Mean Relative Percentage Error (in %)
  pub mrpe: f64,
}
