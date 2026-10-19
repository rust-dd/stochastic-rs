//! `MultivariateExt` — multivariate copula trait.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;

use crate::error::CopulaError;
use crate::multivariate::MultivariateFamily;

/// `cdf`, `pdf` and `log_pdf` answer `MarginalOutOfRange` for a value outside `[0, 1]` or NaN, and `pdf` and `log_pdf`
/// are NaN on the boundary; `fit` answers `MarginalOnBoundary` for a 0 or 1.
pub trait MultivariateExt {
  /// Which family this copula is.
  fn family(&self) -> MultivariateFamily;

  /// `n` rows drawn from `seed`; a fresh `Deterministic` replays them, a shared one advances.
  /// Generic over the seed source, so not callable on a `dyn MultivariateExt`.
  fn sample<S: SeedExt>(&self, n: usize, seed: &S) -> Result<ndarray::Array2<f64>, CopulaError>
  where
    Self: Sized;

  fn fit(&mut self, x: ndarray::ArrayView2<f64>) -> Result<(), CopulaError>;

  fn check_fit(&self, x: ndarray::ArrayView2<f64>) -> Result<(), CopulaError>;

  fn pdf(&self, x: ndarray::ArrayView2<f64>) -> Result<Array1<f64>, CopulaError>;

  fn log_pdf(&self, x: ndarray::ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    Ok(self.pdf(x)?.ln())
  }

  fn cdf(&self, x: ndarray::ArrayView2<f64>) -> Result<Array1<f64>, CopulaError>;
}
