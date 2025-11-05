use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

/// Implements a general GARCH(p,q) model.
///
/// \[
///   \sigma_t^2
///     = \omega
///       + \sum_{i=1}^p \alpha_i \, X_{t-i}^2
///       + \sum_{j=1}^q \beta_j \, \sigma_{t-j}^2,
///   \quad X_t = \sigma_t \, z_t, \quad z_t \sim \mathcal{N}(0,1).
/// \]
///
/// # Parameters
/// - `omega`: Constant term (\(\omega\)) in the GARCH variance equation.
/// - `alpha`: Array \(\{\alpha_1, \ldots, \alpha_p\}\) for past squared observations.
/// - `beta`:  Array \(\{\beta_1, \ldots, \beta_q\}\) for past variances.
/// - `n`:     Length of the time series.
/// - `m`:     Optional batch size (unused by default).
///
/// # Notes
/// 1. Stationarity typically requires \(\sum \alpha_i + \sum \beta_j < 1\).
/// 2. We initialize with an unconditional variance approximation for \(\sigma_0^2\).
#[derive(ImplNew)]
pub struct GARCH<T> {
  pub omega: T,
  pub alpha: Array1<T>,
  pub beta: Array1<T>,
  pub n: usize,
  pub m: Option<usize>,
}

impl SamplingExt<f64> for GARCH<f64> {
  fn sample(&self) -> Array1<f64> {
    let p = self.alpha.len();
    let q = self.beta.len();

    // Generate white noise z_t
    let z = Array1::random(self.n, Normal::new(0.0, 1.0).unwrap());

    // Arrays for X_t and sigma_t^2
    let mut x = Array1::<f64>::zeros(self.n);
    let mut sigma2 = Array1::<f64>::zeros(self.n);

    // Sum of alpha/beta for unconditional variance initialization
    let sum_alpha: f64 = self.alpha.iter().sum();
    let sum_beta: f64 = self.beta.iter().sum();
    let denom = (1.0 - sum_alpha - sum_beta).max(1e-8);

    for t in 0..self.n {
      if t == 0 {
        sigma2[t] = self.omega / denom;
      } else {
        let mut var_t = self.omega;

        // Sum alpha_i * X_{t-i}^2
        for i in 1..=p {
          if t >= i {
            var_t += self.alpha[i - 1] * x[t - i].powi(2);
          }
        }
        // Sum beta_j * sigma2[t-j]
        for j in 1..=q {
          if t >= j {
            var_t += self.beta[j - 1] * sigma2[t - j];
          }
        }
        sigma2[t] = var_t;
      }
      // X_t = sigma_t * z[t]
      x[t] = sigma2[t].sqrt() * z[t];
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

#[cfg(feature = "f32")]
impl SamplingExt<f32> for GARCH<f32> {
  fn sample(&self) -> Array1<f32> {
    let p = self.alpha.len();
    let q = self.beta.len();

    // Generate white noise z_t
    let z = Array1::random(self.n, Normal::new(0.0, 1.0).unwrap()).mapv(|x| x as f32);

    // Arrays for X_t and sigma_t^2
    let mut x = Array1::<f32>::zeros(self.n);
    let mut sigma2 = Array1::<f32>::zeros(self.n);

    // Sum of alpha/beta for unconditional variance initialization
    let sum_alpha: f32 = self.alpha.iter().sum();
    let sum_beta: f32 = self.beta.iter().sum();
    let denom = (1.0 - sum_alpha - sum_beta).max(1e-8);

    for t in 0..self.n {
      if t == 0 {
        sigma2[t] = self.omega / denom;
      } else {
        let mut var_t = self.omega;

        // Sum alpha_i * X_{t-i}^2
        for i in 1..=p {
          if t >= i {
            var_t += self.alpha[i - 1] * x[t - i].powi(2);
          }
        }
        // Sum beta_j * sigma2[t-j]
        for j in 1..=q {
          if t >= j {
            var_t += self.beta[j - 1] * sigma2[t - j];
          }
        }
        sigma2[t] = var_t;
      }
      // X_t = sigma_t * z[t]
      x[t] = sigma2[t].sqrt() * z[t];
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

#[cfg(test)]
mod tests {
  use ndarray::arr1;

  use crate::{
    plot_1d,
    stochastic::{autoregressive::garch::GARCH, SamplingExt},
  };

  #[test]
  fn garch_plot() {
    let alpha = arr1(&[0.05, 0.02]); // p=2
    let beta = arr1(&[0.9]); // q=1
    let garchpq = GARCH::new(0.1, alpha, beta, 100, None);
    plot_1d!(garchpq.sample(), "GARCH(p,q) process");
  }
}
