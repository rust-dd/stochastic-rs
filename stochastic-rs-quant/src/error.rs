//! One error type for quant's data-dependent failures; a broken precondition panics instead.

use std::fmt;

/// Why a quant computation could not produce a result from the data it was given.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum QuantError {
  /// A value carried by the data (a start, a config knob, a model input) lies outside its domain.
  InvalidInput {
    what: &'static str,
    reason: String,
  },
  /// Quote `index` cannot be used (non-finite or non-positive strike, price or maturity, unknown maturity).
  InvalidQuote {
    index: usize,
    reason: String,
  },
  DimensionMismatch {
    what: &'static str,
    expected: usize,
    got: usize,
  },
  InsufficientData {
    what: &'static str,
    needed: usize,
    got: usize,
  },
  /// An input the computation needs was not supplied (a forward for a maturity, a derivative grid).
  MissingInput {
    what: &'static str,
    detail: String,
  },
  NotPositiveDefinite {
    what: &'static str,
  },
  /// The optimiser produced no usable iterate; non-convergence is `Ok(converged = false)`, not this.
  Optimizer {
    what: &'static str,
    reason: String,
  },
  /// Overflow, a non-finite intermediate or a degenerate covariance where a finite number was due.
  Numerical {
    what: &'static str,
    reason: String,
  },
  MarketData {
    symbol: String,
    reason: String,
  },
  Unsupported {
    what: &'static str,
  },
}

impl QuantError {
  pub(crate) fn invalid(what: &'static str, reason: impl Into<String>) -> Self {
    Self::InvalidInput {
      what,
      reason: reason.into(),
    }
  }
}

pub(crate) fn check_grid_shape(
  axes: [&'static str; 2],
  got: (usize, usize),
  expected: (usize, usize),
) -> Result<(), QuantError> {
  for (what, (got, expected)) in axes
    .into_iter()
    .zip([(got.0, expected.0), (got.1, expected.1)])
  {
    if got != expected {
      return Err(QuantError::DimensionMismatch {
        what,
        expected,
        got,
      });
    }
  }
  Ok(())
}

impl fmt::Display for QuantError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::InvalidInput { what, reason } => write!(f, "invalid {what}: {reason}"),
      Self::InvalidQuote { index, reason } => write!(f, "quote {index} cannot be used: {reason}"),
      Self::DimensionMismatch {
        what,
        expected,
        got,
      } => write!(f, "{what}: expected {expected}, got {got}"),
      Self::InsufficientData { what, needed, got } => {
        let plural = if *needed == 1 { "" } else { "s" };
        write!(f, "{what} needs {needed} observation{plural}, got {got}")
      }
      Self::MissingInput { what, detail } => write!(f, "missing {what}: {detail}"),
      Self::NotPositiveDefinite { what } => write!(f, "{what} is not positive definite"),
      Self::Optimizer { what, reason } => {
        write!(f, "{what}: the optimiser produced no result ({reason})")
      }
      Self::Numerical { what, reason } => write!(f, "{what}: {reason}"),
      Self::MarketData { symbol, reason } => write!(f, "market data for {symbol}: {reason}"),
      Self::Unsupported { what } => write!(f, "{what} is not supported"),
    }
  }
}

impl std::error::Error for QuantError {}
