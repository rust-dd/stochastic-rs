use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

/// A generic Asymmetric GARCH(p,q) model (A-GARCH),
/// allowing a separate "delta" term for negative-lag effects.
///
/// One possible form:
/// \[
///   \sigma_t^2
///     = \omega
///       + \sum_{i=1}^p \Bigl[\alpha_i X_{t-i}^2
///                             + \delta_i X_{t-i}^2 \mathbf{1}_{\{X_{t-i}<0\}}\Bigr]
///       + \sum_{j=1}^q \beta_j \sigma_{t-j}^2,
///   \quad X_t = \sigma_t \cdot z_t, \quad z_t \sim \mathcal{N}(0,1).
/// \]
///
/// # Parameters
/// - `omega`: Constant term \(\omega\).
/// - `alpha`: Array \(\{\alpha_1, \ldots, \alpha_p\}\) for positive squared terms.
/// - `delta`: Array \(\{\delta_1, \ldots, \delta_p\}\) for negative-lag extra effect.
/// - `beta`:  Array \(\{\beta_1, \ldots, \beta_q\}\).
/// - `n`:     Length of the time series.
/// - `m`:     Optional batch size (unused by default).
///
/// # Notes
/// - This is essentially a T-GARCH-like structure but with different naming (`delta`).
/// - Stationarity constraints typically require \(\sum \alpha_i + \tfrac{1}{2}\sum \delta_i + \sum \beta_j < 1\).
#[derive(ImplNew)]
pub struct AGARCH<T> {
  pub omega: T,
  pub alpha: Array1<T>,
  pub delta: Array1<T>,
  pub beta: Array1<T>,
  pub n: usize,
  pub m: Option<usize>,
}

#[cfg(feature = "f64")]
impl SamplingExt<f64> for AGARCH<f64> {
  fn sample(&self) -> Array1<f64> {
    let p = self.alpha.len();
    let q = self.beta.len();

    // Generate white noise
    let z = Array1::random(self.n, Normal::new(0.0, 1.0).unwrap());

    // Arrays for X_t and sigma_t^2
    let mut x = Array1::<f64>::zeros(self.n);
    let mut sigma2 = Array1::<f64>::zeros(self.n);

    // Summation for unconditional variance init
    let sum_alpha: f64 = self.alpha.iter().sum();
    let sum_delta_half: f64 = self.delta.iter().sum::<f64>() * 0.5;
    let sum_beta: f64 = self.beta.iter().sum();
    let denom = (1.0 - sum_alpha - sum_delta_half - sum_beta).max(1e-8);

    for t in 0..self.n {
      if t == 0 {
        sigma2[t] = self.omega / denom;
      } else {
        let mut var_t = self.omega;
        // p-lag terms
        for i in 1..=p {
          if t >= i {
            let x_lag = x[t - i];
            let indicator = if x_lag < 0.0 { 1.0 } else { 0.0 };

            var_t +=
              self.alpha[i - 1] * x_lag.powi(2) + self.delta[i - 1] * x_lag.powi(2) * indicator;
          }
        }
        // q-lag terms
        for j in 1..=q {
          if t >= j {
            var_t += self.beta[j - 1] * sigma2[t - j];
          }
        }
        sigma2[t] = var_t;
      }
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
impl SamplingExt<f32> for AGARCH<f32> {
  fn sample(&self) -> Array1<f32> {
    let p = self.alpha.len();
    let q = self.beta.len();

    // Generate white noise
    let z = Array1::random(self.n, Normal::new(0.0, 1.0).unwrap()).mapv(|x| x as f32);

    // Arrays for X_t and sigma_t^2
    let mut x = Array1::<f32>::zeros(self.n);
    let mut sigma2 = Array1::<f32>::zeros(self.n);

    // Summation for unconditional variance init
    let sum_alpha: f32 = self.alpha.iter().sum();
    let sum_delta_half: f32 = self.delta.iter().sum::<f32>() * 0.5;
    let sum_beta: f32 = self.beta.iter().sum();
    let denom = (1.0 - sum_alpha - sum_delta_half - sum_beta).max(1e-8);

    for t in 0..self.n {
      if t == 0 {
        sigma2[t] = self.omega / denom;
      } else {
        let mut var_t = self.omega;
        // p-lag terms
        for i in 1..=p {
          if t >= i {
            let x_lag = x[t - i];
            let indicator = if x_lag < 0.0 { 1.0 } else { 0.0 };

            var_t +=
              self.alpha[i - 1] * x_lag.powi(2) + self.delta[i - 1] * x_lag.powi(2) * indicator;
          }
        }
        // q-lag terms
        for j in 1..=q {
          if t >= j {
            var_t += self.beta[j - 1] * sigma2[t - j];
          }
        }
        sigma2[t] = var_t;
      }
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
    stochastic::{autoregressive::agrach::AGARCH, SamplingExt},
  };

  #[test]
  fn agarch_plot() {
    let alpha = arr1(&[0.05, 0.01]); // p=2
    let delta = arr1(&[0.03, 0.01]); // p=2
    let beta = arr1(&[0.8]); // q=1
    let agarchpq = AGARCH::new(0.1, alpha, delta, beta, 100, None);
    plot_1d!(agarchpq.sample(), "A-GARCH(p,q) process");
  }
}
