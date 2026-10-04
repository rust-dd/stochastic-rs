//! The one error type of the crate; `#[non_exhaustive]`, `Send + Sync`, so it crosses threads and `?` into `anyhow`.

use std::fmt;

/// Why a copula operation could not produce a value.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum CopulaError {
  /// A parameter outside its domain; `constraint` is the predicate it failed, in the crate's assert form.
  InvalidParameter {
    name: &'static str,
    value: f64,
    constraint: String,
  },
  /// The copula was queried before `fit`, `set_theta` or a correlation matrix gave it parameters.
  NotFitted,
  /// The data's column count (or a tree's leaf count) does not match the copula's dimension.
  DimensionMismatch { expected: usize, got: usize },
  /// Fewer observations than the fit needs.
  InsufficientData { needed: usize, got: usize },
  /// A pseudo-observation outside `[0, 1]`.
  MarginalOutOfRange,
  /// The pseudo-observations fail the uniformity check a copula fit assumes.
  MarginalNotUniform,
  /// Not well-formed: a correlation matrix, a vine or NAC tree, or fewer than two dimensions.
  InvalidStructure(String),
  /// The family has no such operation (an Archimedean generator, an unimplemented fit).
  Unsupported(String),
  /// A numerical inversion did not converge.
  Numerical(String),
}

impl fmt::Display for CopulaError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      CopulaError::InvalidParameter {
        name,
        value,
        constraint,
      } => write!(
        f,
        "{name} must satisfy `{constraint}`, got {name} = {value:?}"
      ),
      CopulaError::NotFitted => {
        f.write_str("the copula has no parameters yet: fit it or set them first")
      }
      CopulaError::DimensionMismatch { expected, got } => {
        write!(
          f,
          "dimension mismatch: the copula has dim {expected}, the input has {got}"
        )
      }
      CopulaError::InsufficientData { needed, got } => {
        write!(
          f,
          "too little data: {got} observations, at least {needed} needed"
        )
      }
      CopulaError::MarginalOutOfRange => f.write_str("marginal values must lie in [0, 1]"),
      CopulaError::MarginalNotUniform => {
        f.write_str("marginal values do not follow a uniform distribution")
      }
      CopulaError::InvalidStructure(why) => write!(f, "invalid structure: {why}"),
      CopulaError::Unsupported(what) => write!(f, "{what}"),
      CopulaError::Numerical(why) => write!(f, "numerical failure: {why}"),
    }
  }
}

impl std::error::Error for CopulaError {}
