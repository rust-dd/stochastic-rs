//! One error type for every failure the crate returns.

use std::fmt;

/// Why an estimator or test could not answer; a validated precondition panics instead.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum StatsError {
  /// Fewer observations, series or bins than the method needs.
  InsufficientData { needed: usize, got: usize },
  /// Two inputs that must agree in length or shape do not.
  DimensionMismatch { expected: usize, got: usize },
  /// An observation outside the input's domain: non-finite, or non-positive where positivity is needed.
  InvalidObservation {
    index: usize,
    value: f64,
    constraint: &'static str,
  },
  /// The data carry nothing the method can use: a constant series, zero variance, no valid scales.
  DegenerateData(&'static str),
  /// A configuration value outside its range, for estimators built by struct literal and checked at `estimate`.
  InvalidParameter {
    name: &'static str,
    value: f64,
    constraint: &'static str,
  },
  /// A regression or linear system without a finite solution.
  Singular(&'static str),
  /// An optimiser or filter that produced no usable value at `stage`.
  Numerical { stage: &'static str },
}

impl fmt::Display for StatsError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::InsufficientData { needed, got } => {
        write!(f, "needs at least {needed}, got {got}")
      }
      Self::DimensionMismatch { expected, got } => {
        write!(f, "expected {expected} entries, got {got}")
      }
      Self::InvalidObservation {
        index,
        value,
        constraint,
      } => {
        write!(
          f,
          "observation {index} = {value:?} must satisfy `{constraint}`"
        )
      }
      Self::DegenerateData(what) => write!(f, "{what}"),
      Self::InvalidParameter {
        name,
        value,
        constraint,
      } => {
        write!(
          f,
          "{name} must satisfy `{constraint}`, got {name} = {value:?}"
        )
      }
      Self::Singular(what) => write!(f, "{what} has no finite solution"),
      Self::Numerical { stage } => write!(f, "{stage} produced no usable value"),
    }
  }
}

impl std::error::Error for StatsError {}
