use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

/// Implements a general T-GARCH (GJR-GARCH)(p,q) model:
///
/// \[
///   \sigma_t^2
///     = \omega
///       + \sum_{i=1}^p \Bigl[\alpha_i X_{t-i}^2
///                              + \gamma_i X_{t-i}^2 \mathbf{1}_{\{X_{t-i}<0\}}\Bigr]
///       + \sum_{j=1}^q \beta_j \sigma_{t-j}^2,
///   \quad X_t = \sigma_t \cdot z_t, \quad z_t \sim \mathcal{N}(0,1).
/// \]
///
/// # Parameters
/// - `omega`: Constant term (\(\omega\)).
/// - `alpha`: Array \(\{\alpha_1, \ldots, \alpha_p\}\) for the positive part of squared residuals.
/// - `gamma`: Array \(\{\gamma_1, \ldots, \gamma_p\}\) for the threshold effect (negative residuals).
///            Must have the same length as `alpha`.
/// - `beta`:  Array \(\{\beta_1, \ldots, \beta_q\}\) for the past variance terms.
/// - `n`:     Length of the time series to generate.
/// - `m`:     Optional batch size (unused by default).
///
/// # Notes
/// - Stationarity constraints typically include: \(\sum \alpha_i + \tfrac{1}{2}\sum \gamma_i + \sum \beta_j < 1\).
/// - We do a simple unconditional variance initialization for \(\sigma_0^2\).
#[derive(ImplNew)]
pub struct TGARCH<T> {
  pub omega: T,
  pub alpha: Array1<T>,
  pub gamma: Array1<T>,
  pub beta: Array1<T>,
  pub n: usize,
  pub m: Option<usize>,
}

#[cfg(feature = "f64")]
impl SamplingExt<f64> for TGARCH<f64> {
  fn sample(&self) -> Array1<f64> {
    let p = self.alpha.len();
    let q = self.beta.len();

    // Standard normal noise
    let z = Array1::random(self.n, Normal::new(0.0, 1.0).unwrap());

    // Arrays for X_t and sigma_t^2
    let mut x = Array1::<f64>::zeros(self.n);
    let mut sigma2 = Array1::<f64>::zeros(self.n);

    // Sum up alpha + 0.5 gamma + beta for unconditional variance approximation
    let sum_alpha: f64 = self.alpha.iter().sum();
    let sum_gamma_half: f64 = self.gamma.iter().sum::<f64>() * 0.5;
    let sum_beta: f64 = self.beta.iter().sum();
    let denom = (1.0 - sum_alpha - sum_gamma_half - sum_beta).max(1e-8);

    for t in 0..self.n {
      if t == 0 {
        sigma2[t] = self.omega / denom;
      } else {
        let mut var_t = self.omega;

        // Sum over p lags
        for i in 1..=p {
          if t >= i {
            let x_lag = x[t - i];
            // Threshold indicator
            let indicator = if x_lag < 0.0 { 1.0 } else { 0.0 };

            // alpha_i * X_{t-i}^2 + gamma_i * X_{t-i}^2 * indicator
            var_t +=
              self.alpha[i - 1] * x_lag.powi(2) + self.gamma[i - 1] * x_lag.powi(2) * indicator;
          }
        }

        // Sum over q lags
        for j in 1..=q {
          if t >= j {
            var_t += self.beta[j - 1] * sigma2[t - j];
          }
        }

        sigma2[t] = var_t;
      }
      // X_t = sigma_t * z_t
      x[t] = sigma2[t].sqrt() * z[t];
    }

    x
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f64> {
    use crate::stats::distr::normal::SimdNormal;

    let p = self.alpha.len();
    let q = self.beta.len();

    // Standard normal noise
    let z = Array1::random(self.n, SimdNormal::new(0.0, 1.0));

    // Arrays for X_t and sigma_t^2
    let mut x = Array1::<f64>::zeros(self.n);
    let mut sigma2 = Array1::<f64>::zeros(self.n);

    // Sum up alpha + 0.5 gamma + beta for unconditional variance approximation
    let sum_alpha: f64 = self.alpha.iter().sum();
    let sum_gamma_half: f64 = self.gamma.iter().sum::<f64>() * 0.5;
    let sum_beta: f64 = self.beta.iter().sum();
    let denom = (1.0 - sum_alpha - sum_gamma_half - sum_beta).max(1e-8);

    for t in 0..self.n {
      if t == 0 {
        sigma2[t] = self.omega / denom;
      } else {
        let mut var_t = self.omega;

        // Sum over p lags
        for i in 1..=p {
          if t >= i {
            let x_lag = x[t - i];
            // Threshold indicator
            let indicator = if x_lag < 0.0 { 1.0 } else { 0.0 };

            // alpha_i * X_{t-i}^2 + gamma_i * X_{t-i}^2 * indicator
            var_t +=
              self.alpha[i - 1] * x_lag.powi(2) + self.gamma[i - 1] * x_lag.powi(2) * indicator;
          }
        }

        // Sum over q lags
        for j in 1..=q {
          if t >= j {
            var_t += self.beta[j - 1] * sigma2[t - j];
          }
        }

        sigma2[t] = var_t;
      }
      // X_t = sigma_t * z_t
      x[t] = sigma2[t].sqrt() * z[t] as f64;
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
impl SamplingExt<f32> for TGARCH<f32> {
  fn sample(&self) -> Array1<f32> {
    let p = self.alpha.len();
    let q = self.beta.len();

    // Standard normal noise
    let z = Array1::random(self.n, Normal::new(0.0, 1.0).unwrap()).mapv(|x| x as f32);

    // Arrays for X_t and sigma_t^2
    let mut x = Array1::<f32>::zeros(self.n);
    let mut sigma2 = Array1::<f32>::zeros(self.n);

    // Sum up alpha + 0.5 gamma + beta for unconditional variance approximation
    let sum_alpha: f32 = self.alpha.iter().sum();
    let sum_gamma_half: f32 = self.gamma.iter().sum::<f32>() * 0.5;
    let sum_beta: f32 = self.beta.iter().sum();
    let denom = (1.0 - sum_alpha - sum_gamma_half - sum_beta).max(1e-8);

    for t in 0..self.n {
      if t == 0 {
        sigma2[t] = self.omega / denom;
      } else {
        let mut var_t = self.omega;

        // Sum over p lags
        for i in 1..=p {
          if t >= i {
            let x_lag = x[t - i];
            // Threshold indicator
            let indicator = if x_lag < 0.0 { 1.0 } else { 0.0 };

            // alpha_i * X_{t-i}^2 + gamma_i * X_{t-i}^2 * indicator
            var_t +=
              self.alpha[i - 1] * x_lag.powi(2) + self.gamma[i - 1] * x_lag.powi(2) * indicator;
          }
        }

        // Sum over q lags
        for j in 1..=q {
          if t >= j {
            var_t += self.beta[j - 1] * sigma2[t - j];
          }
        }

        sigma2[t] = var_t;
      }
      // X_t = sigma_t * z_t
      x[t] = sigma2[t].sqrt() * z[t];
    }

    x
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f32> {
    use crate::stats::distr::normal::SimdNormal;

    let p = self.alpha.len();
    let q = self.beta.len();

    // Standard normal noise
    let z = Array1::random(self.n, SimdNormal::new(0.0, 1.0));

    // Arrays for X_t and sigma_t^2
    let mut x = Array1::<f32>::zeros(self.n);
    let mut sigma2 = Array1::<f32>::zeros(self.n);

    // Sum up alpha + 0.5 gamma + beta for unconditional variance approximation
    let sum_alpha: f32 = self.alpha.iter().sum();
    let sum_gamma_half: f32 = self.gamma.iter().sum::<f32>() * 0.5;
    let sum_beta: f32 = self.beta.iter().sum();
    let denom = (1.0 - sum_alpha - sum_gamma_half - sum_beta).max(1e-8);

    for t in 0..self.n {
      if t == 0 {
        sigma2[t] = self.omega / denom;
      } else {
        let mut var_t = self.omega;

        // Sum over p lags
        for i in 1..=p {
          if t >= i {
            let x_lag = x[t - i];
            // Threshold indicator
            let indicator = if x_lag < 0.0 { 1.0 } else { 0.0 };

            // alpha_i * X_{t-i}^2 + gamma_i * X_{t-i}^2 * indicator
            var_t +=
              self.alpha[i - 1] * x_lag.powi(2) + self.gamma[i - 1] * x_lag.powi(2) * indicator;
          }
        }

        // Sum over q lags
        for j in 1..=q {
          if t >= j {
            var_t += self.beta[j - 1] * sigma2[t - j];
          }
        }

        sigma2[t] = var_t;
      }
      // X_t = sigma_t * z_t
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
    stochastic::{autoregressive::tgarch::TGARCH, SamplingExt},
  };

  fn tgarchpq_plot() {
    let alpha = arr1(&[0.05, 0.01]); // p=2
    let gamma = arr1(&[0.02, 0.01]); // p=2
    let beta = arr1(&[0.9]); // q=1
    let tgarchpq = TGARCH::new(0.1, alpha, gamma, beta, 100, None);
    plot_1d!(tgarchpq.sample(), "T-GARCH(p,q) process");
  }
}
