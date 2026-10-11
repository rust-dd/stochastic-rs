//! `MultivariateExt` — feature-gated multivariate copula trait.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;

use crate::error::CopulaError;
use crate::multivariate::CopulaType as MultivariateCopulaType;

pub trait MultivariateExt {
  fn r#type(&self) -> MultivariateCopulaType;

  /// `n` rows drawn from `seed`; a `Deterministic` seed replays them.
  /// Generic over the seed source, so not callable on a `dyn MultivariateExt`.
  fn sample<S: SeedExt>(&self, n: usize, seed: &S) -> Result<ndarray::Array2<f64>, CopulaError>
  where
    Self: Sized;

  fn fit(&mut self, X: ndarray::ArrayView2<f64>) -> Result<(), CopulaError>;

  fn check_fit(&self, X: ndarray::ArrayView2<f64>) -> Result<(), CopulaError>;

  fn pdf(&self, X: ndarray::ArrayView2<f64>) -> Result<Array1<f64>, CopulaError>;

  fn log_pdf(&self, X: ndarray::ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    Ok(self.pdf(X)?.ln())
  }

  fn cdf(&self, X: ndarray::ArrayView2<f64>) -> Result<Array1<f64>, CopulaError>;
}
