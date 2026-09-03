//! Scalar Gaussian QML for latent log variance observed through block RV.
//!
//! For a block containing `m` conditionally Gaussian returns, annualised
//! realised variance satisfies `RV / exp(h) ~ chi_squared(m) / m`. Hence
//! `ln(RV) - b_m = h + epsilon`, where
//! `b_m = digamma(m / 2) - ln(m / 2)` and
//! `Var(epsilon) = trigamma(m / 2)`. The latent state follows
//! `h[t+1] = mu + phi * (h[t] - mu) + eta[t+1]` with
//! `Var(eta) = q`.
//!
//! The Gaussian measurement approximation and Kalman quasi-likelihood are
//! the scalar block-realised-variance analogue of the stochastic-variance
//! state-space treatment in Harvey, Ruiz, and Shephard (1994), "Multivariate
//! Stochastic Variance Models", *Review of Economic Studies* 61(2), 247-264,
//! <https://doi.org/10.2307/2297980>.
//! Parameter uncertainty uses natural-parameter numerical scores and White's
//! `H^-1 B H^-1` QML sandwich covariance; see White (1982), "Maximum
//! Likelihood Estimation of Misspecified Models", *Econometrica* 50(1), 1-25,
//! <https://doi.org/10.2307/1912526>.

use std::error::Error;
use std::fmt;

mod filter;
mod fit;
mod math;
mod uncertainty;

pub use filter::LogRealizedVarianceFilterResult;
pub use filter::filter_log_realized_variance;
pub use fit::fit_log_realized_variance_qml;

/// Exact mean and variance of `ln(chi_squared(m) / m)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogChiSquareMoments {
  /// Bias `E[ln(chi_squared(m) / m)]` subtracted from log RV.
  pub log_bias: f64,
  /// Measurement variance `Var[ln(chi_squared(m) / m)]`.
  pub log_variance: f64,
}

/// Returns the log-chi-square measurement moments for a positive block size.
pub fn log_chi_square_moments(
  block_degrees_of_freedom: usize,
) -> Result<LogChiSquareMoments, LogRealizedVarianceQmlError> {
  math::measurement_moments(block_degrees_of_freedom)
}

/// Structural bounds imposed during deterministic QML optimisation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogRealizedVarianceQmlBounds {
  /// Smallest admissible stationary mean log variance.
  pub min_mu: f64,
  /// Largest admissible stationary mean log variance.
  pub max_mu: f64,
  /// Smallest admissible stationary AR coefficient, strictly above `-1`.
  pub min_phi: f64,
  /// Largest admissible stationary AR coefficient, strictly below `1`.
  pub max_phi: f64,
  /// Smallest admissible positive latent innovation variance.
  pub min_q: f64,
  /// Largest admissible latent innovation variance.
  pub max_q: f64,
}

impl Default for LogRealizedVarianceQmlBounds {
  fn default() -> Self {
    Self {
      min_mu: -20.0,
      max_mu: 5.0,
      min_phi: -0.99,
      max_phi: 0.999,
      min_q: 1e-8,
      max_q: 4.0,
    }
  }
}

/// Deterministic fit controls and structural parameter bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogRealizedVarianceQmlConfig {
  /// Effective chi-square degrees of freedom in each non-overlapping RV block.
  pub block_degrees_of_freedom: usize,
  /// Minimum number of RV blocks required by the fit.
  pub minimum_observations: usize,
  /// Iteration cap applied independently to every deterministic start.
  pub max_iterations_per_start: usize,
  /// Structural parameter bounds used by the constrained transform.
  pub bounds: LogRealizedVarianceQmlBounds,
}

impl Default for LogRealizedVarianceQmlConfig {
  fn default() -> Self {
    Self {
      block_degrees_of_freedom: 5,
      minimum_observations: 32,
      max_iterations_per_start: 1_000,
      bounds: LogRealizedVarianceQmlBounds::default(),
    }
  }
}

/// Latent log-variance state parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogRealizedVarianceParameters {
  /// Stationary mean of latent log variance.
  pub mu: f64,
  /// Stationary AR(1) persistence in `(-1, 1)`.
  pub phi: f64,
  /// Positive latent log-state innovation variance.
  pub q: f64,
}

/// Deterministic multi-start optimiser and innovation diagnostics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogRealizedVarianceQmlDiagnostics {
  /// Exact log-chi-square bias removed from every observation.
  pub log_measurement_bias: f64,
  /// Exact log-chi-square measurement variance used by the Kalman filter.
  pub log_measurement_variance: f64,
  /// Root mean square of the one-step log-measurement innovations.
  pub innovation_rmse: f64,
  /// Maximised Gaussian Kalman quasi-log-likelihood.
  pub log_likelihood: f64,
  /// Number of deterministic Nelder-Mead starts evaluated.
  pub starts_attempted: usize,
  /// Number of valid starts satisfying the optimiser tolerance.
  pub converged_starts: usize,
  /// Zero-based index of the selected deterministic start.
  pub selected_start_index: usize,
  /// Nelder-Mead iterations used by the selected start.
  pub selected_iterations: usize,
  /// Sum of iterations over all deterministic starts.
  pub total_iterations: usize,
  /// Whether the selected maximum-likelihood start converged.
  pub converged: bool,
}

/// Structural-bound proximity flags in natural parameter coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LogRealizedVarianceParameterBoundaryFlags {
  pub mu_at_lower_bound: bool,
  pub mu_at_upper_bound: bool,
  pub phi_at_lower_bound: bool,
  pub phi_at_upper_bound: bool,
  pub q_at_lower_bound: bool,
  pub q_at_upper_bound: bool,
}

impl LogRealizedVarianceParameterBoundaryFlags {
  /// Whether any fitted parameter is within the structural boundary tolerance.
  pub fn any(self) -> bool {
    self.mu_at_lower_bound
      || self.mu_at_upper_bound
      || self.phi_at_lower_bound
      || self.phi_at_upper_bound
      || self.q_at_lower_bound
      || self.q_at_upper_bound
  }
}

/// Numerical observed-information uncertainty in `[mu, phi, q]` order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogRealizedVarianceParameterUncertainty {
  /// Central-difference observed information, absent if it is not finite.
  pub observed_information: Option<[[f64; 3]; 3]>,
  /// Sum of per-observation natural-parameter score outer products.
  pub score_outer_product: Option<[[f64; 3]; 3]>,
  /// Inverse observed-information covariance, absent when singular.
  pub observed_information_covariance: Option<[[f64; 3]; 3]>,
  /// Robust `H^-1 B H^-1` QML covariance, absent when unusable.
  pub robust_sandwich_covariance: Option<[[f64; 3]; 3]>,
  /// Robust natural-parameter standard errors in `[mu, phi, q]` order.
  pub robust_standard_errors: Option<[f64; 3]>,
  /// Infinity-norm condition estimate after diagonal information scaling.
  pub scaled_condition_number: f64,
  /// True when the scaled observed information is not positive definite.
  pub singular: bool,
  /// True when the scaled condition number exceeds the numerical quality cap.
  pub ill_conditioned: bool,
  /// True when the robust sandwich covariance is finite and positive definite.
  pub robust_covariance_usable: bool,
  /// Proximity to every configured structural parameter bound.
  pub boundary: LogRealizedVarianceParameterBoundaryFlags,
}

/// Statistical quality gates for accepting a fitted model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogRealizedVarianceQmlQuality {
  /// The selected deterministic Nelder-Mead start converged.
  pub optimizer_converged: bool,
  /// No parameter is within the structural boundary tolerance.
  pub parameters_interior: bool,
  /// The numerical observed information is positive definite.
  pub observed_information_nonsingular: bool,
  /// The scaled information condition number passes its numerical cap.
  pub observed_information_well_conditioned: bool,
  /// The robust sandwich covariance is finite and positive definite.
  pub robust_covariance_usable: bool,
  /// All quality gates pass.
  pub accepted: bool,
}

/// Fitted state, causal filter paths, and arithmetic-variance forecasts.
#[derive(Clone, Debug, PartialEq)]
pub struct LogRealizedVarianceQmlResult {
  /// Selected QML parameters.
  pub parameters: LogRealizedVarianceParameters,
  /// Final causal posterior mean of latent log variance.
  pub current_filtered_log_variance: f64,
  /// Final causal posterior covariance of latent log variance.
  pub current_filtered_state_covariance: f64,
  /// Causal posterior log-state means aligned with the RV observations.
  pub filtered_log_variance_path: Vec<f64>,
  /// Causal posterior covariances aligned with the RV observations.
  pub filtered_state_covariance_path: Vec<f64>,
  /// Posterior `E[exp(h_t) | RV_0, ..., RV_t]`.
  pub current_arithmetic_variance: f64,
  /// Unconditional stationary `E[exp(h)]` under the fitted AR(1).
  pub stationary_arithmetic_variance: f64,
  /// Conditional one-step-ahead `E[exp(h_{t+1}) | RV_0, ..., RV_t]`.
  pub one_step_arithmetic_variance_forecast: f64,
  /// Numerical observed-information parameter uncertainty.
  pub parameter_uncertainty: LogRealizedVarianceParameterUncertainty,
  /// Structural, optimiser, and identification quality gates.
  pub quality: LogRealizedVarianceQmlQuality,
  /// Likelihood, measurement, and optimiser audit values.
  pub diagnostics: LogRealizedVarianceQmlDiagnostics,
}

/// Typed validation, optimisation, and numerical failures.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LogRealizedVarianceQmlError {
  InsufficientObservations { actual: usize, minimum: usize },
  NonFiniteObservation { index: usize },
  NonPositiveObservation { index: usize },
  InvalidConfig { field: &'static str },
  InvalidParameter { field: &'static str },
  OptimizationFailed,
  NumericalFailure { stage: &'static str, index: usize },
}

impl fmt::Display for LogRealizedVarianceQmlError {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::InsufficientObservations { actual, minimum } => write!(
        formatter,
        "log-realized-variance QML requires at least {minimum} observations, got {actual}"
      ),
      Self::NonFiniteObservation { index } => {
        write!(
          formatter,
          "realized variance at index {index} is not finite"
        )
      }
      Self::NonPositiveObservation { index } => {
        write!(
          formatter,
          "realized variance at index {index} is not positive"
        )
      }
      Self::InvalidConfig { field } => {
        write!(
          formatter,
          "invalid log-realized-variance QML config field: {field}"
        )
      }
      Self::InvalidParameter { field } => {
        write!(
          formatter,
          "invalid log-realized-variance parameter: {field}"
        )
      }
      Self::OptimizationFailed => {
        write!(formatter, "all log-realized-variance QML starts failed")
      }
      Self::NumericalFailure { stage, index } => write!(
        formatter,
        "non-finite log-realized-variance value during {stage} at index {index}"
      ),
    }
  }
}

impl Error for LogRealizedVarianceQmlError {}

#[cfg(test)]
#[path = "log_realized_variance_qml/tests.rs"]
mod tests;
