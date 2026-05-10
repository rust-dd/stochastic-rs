//! # Factor Models & Statistical Arbitrage
//!
//! Principal-component factor extraction, well-conditioned covariance
//! estimation via Ledoit-Wolf shrinkage, two-pass Fama-MacBeth cross-sectional
//! regression, and a self-contained cointegrated pairs-trading framework.
//!
//! $$
//! \hat\Sigma_{LW} = \alpha\,\bar s\,I + (1-\alpha)\,S,\qquad
//! \alpha = \min\!\bigl(1,\,b^2 / d^2\bigr).
//! $$
//!
//! Requires the `openblas` feature for SVD / eigendecomposition.
//!
//! # References
//! - Ledoit, Wolf, "A Well-Conditioned Estimator for Large-Dimensional
//!   Covariance Matrices", Journal of Multivariate Analysis, 88(2), 365-411
//!   (2004). DOI: 10.1016/S0047-259X(03)00096-4
//! - Ledoit, Wolf, "Honey, I Shrunk the Sample Covariance Matrix", Journal of
//!   Portfolio Management, 30(4), 110-119 (2004).
//!   DOI: 10.3905/jpm.2004.110
//! - Fama, MacBeth, "Risk, Return, and Equilibrium: Empirical Tests", Journal
//!   of Political Economy, 81(3), 607-636 (1973). DOI: 10.1086/260061
//! - Engle, Granger, "Co-Integration and Error Correction: Representation,
//!   Estimation, and Testing", Econometrica, 55(2), 251-276 (1987).
//!   DOI: 10.2307/1913236
//! - Gatev, Goetzmann, Rouwenhorst, "Pairs Trading: Performance of a
//!   Relative-Value Arbitrage Rule", Review of Financial Studies, 19(3),
//!   797-827 (2006). DOI: 10.1093/rfs/hhj020

#[cfg(feature = "openblas")]
pub mod fama_macbeth;
#[cfg(feature = "openblas")]
pub mod pairs;
#[cfg(feature = "openblas")]
pub mod pca;
pub mod shrinkage;

/// Error type returned by `try_*` variants of factor-model routines that may
/// fail on rank-deficient inputs (pure noise, perfectly collinear columns,
/// SVD non-convergence).
#[derive(Debug)]
pub enum FactorsError {
  /// SVD on a numerical input failed to converge.
  SvdFailed(String),
  /// Ordinary-least-squares (linear regression) input was singular.
  OlsFailed(String),
  /// Generic numerical fault from a downstream linear-algebra routine.
  Numerical(String),
}

impl std::fmt::Display for FactorsError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::SvdFailed(msg) => write!(f, "SVD failed: {msg}"),
      Self::OlsFailed(msg) => write!(f, "OLS failed: {msg}"),
      Self::Numerical(msg) => write!(f, "numerical fault: {msg}"),
    }
  }
}

impl std::error::Error for FactorsError {}

#[cfg(feature = "openblas")]
pub use fama_macbeth::FamaMacBethResult;
#[cfg(feature = "openblas")]
pub use fama_macbeth::fama_macbeth;
#[cfg(feature = "openblas")]
pub use pairs::PairsSignal;
#[cfg(feature = "openblas")]
pub use pairs::PairsStrategy;
#[cfg(feature = "openblas")]
pub use pairs::pairs_signals;
#[cfg(feature = "openblas")]
pub use pca::PcaResult;
#[cfg(feature = "openblas")]
pub use pca::pca_decompose;
#[cfg(feature = "openblas")]
pub use pca::try_pca_decompose;
pub use shrinkage::ledoit_wolf_shrinkage;
pub use shrinkage::sample_covariance;
