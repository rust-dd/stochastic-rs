use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::Sampling;

/// Implements an ARCH(m) model:
///
/// \[
///   \sigma_t^2 = \omega + \sum_{i=1}^m \alpha_i X_{t-i}^2,
///   \quad X_t = \sigma_t \cdot z_t, \quad z_t \sim \mathcal{N}(0,1).
/// \]
///
/// # Fields
/// - `omega`: Constant term.
/// - `alpha`: Array of ARCH coefficients.
/// - `n`: Number of observations.
/// - `m`: Optional batch size.
#[derive(ImplNew)]
pub struct ARCH {
  /// Omega (constant term in variance)
  pub omega: f64,
  /// Coefficients alpha_i
  pub alpha: Array1<f64>,
  /// Length of series
  pub n: usize,
  /// Optional batch size
  pub m: Option<usize>,
}

impl Sampling<f64> for ARCH {
  fn sample(&self) -> Array1<f64> {
    let m = self.alpha.len();
    let z = Array1::random(self.n, Normal::new(0.0, 1.0).unwrap());
    let mut x = Array1::<f64>::zeros(self.n);

    for t in 0..self.n {
      // compute sigma_t^2
      let mut var_t = self.omega;
      for i in 1..=m {
        if t >= i {
          let x_lag = x[t - i];
          var_t += self.alpha[i - 1] * x_lag.powi(2);
        }
      }
      let sigma_t = var_t.sqrt();
      x[t] = sigma_t * z[t];
    }

    x
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
