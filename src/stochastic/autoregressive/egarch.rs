use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

/// Implements an EGARCH(p,q) model:
///
/// \[
///   \ln(\sigma_t^2)
///     = \omega
///       + \sum_{i=1}^p \Bigl[\alpha_i \bigl(\lvert z_{t-i}\rvert - E\lvert z\rvert\bigr)
///                            + \gamma_i \, z_{t-i}\Bigr]
///       + \sum_{j=1}^q \beta_j \,\ln(\sigma_{t-j}^2),
///   \quad X_t = \sigma_t \cdot z_t,\quad z_t \sim \mathcal{N}(0,1).
/// \]
///
/// where
///
/// - \( z_{t-i} = \frac{X_{t-i}}{\sigma_{t-i}} \) is the standardized residual,
/// - \( E\lvert z\rvert = \sqrt{2/\pi} \) under a standard normal assumption.
///
/// # Parameters
/// - `omega`: The constant term \(\omega\) in the log-variance equation.
/// - `alpha`: An array \(\{\alpha_1, \dots, \alpha_p\}\) controlling the magnitude effect.
/// - `gamma`: An array \(\{\gamma_1, \dots, \gamma_p\}\) for the sign (asymmetry) effect.
///            Must be the same length as `alpha`.
/// - `beta`:  An array \(\{\beta_1, \dots, \beta_q\}\) controlling persistence of past log-variance.
/// - `n`: The number of observations to generate.
/// - `m`: Optional batch size for parallel sampling (unused by default).
///
/// # Notes
/// 1. We assume that `alpha` and `gamma` each have length \(p\).
/// 2. We assume that `beta` has length \(q\).
/// 3. Real-world usage typically enforces constraints to ensure stationarity/ergodicity.
#[derive(ImplNew)]
pub struct EGARCH<T> {
  /// Constant term (\(\omega\)) in log-variance
  pub omega: T,
  /// Magnitude effect coefficients (\(\alpha_1, \ldots, \alpha_p\))
  pub alpha: Array1<T>,
  /// Sign/asymmetry effect coefficients (\(\gamma_1, \ldots, \gamma_p\))
  pub gamma: Array1<T>,
  /// Persistence coefficients for log-variance (\(\beta_1, \ldots, \beta_q\))
  pub beta: Array1<T>,
  /// Number of observations
  pub n: usize,
  /// Optional batch size (unused by default)
  pub m: Option<usize>,
}

impl SamplingExt<f64> for EGARCH<f64> {
  fn sample(&self) -> Array1<f64> {
    let p = self.alpha.len();
    let q = self.beta.len();

    // Generate white noise z_t ~ N(0,1)
    let z = Array1::random(self.n, Normal::new(0.0, 1.0).unwrap());

    // Allocate arrays for the time series (X_t) and log of variance (log_sigma2)
    let mut x = Array1::<f64>::zeros(self.n);
    let mut log_sigma2 = Array1::<f64>::zeros(self.n);

    // For normal(0,1), the expected absolute value is sqrt(2/pi)
    let e_abs_z = (2.0 / std::f64::consts::PI).sqrt();

    for t in 0..self.n {
      if t == 0 {
        // Initialize log-variance (e.g., with omega)
        log_sigma2[t] = self.omega;
      } else {
        // 1) Compute the shock term from p lags
        let mut shock_term = 0.0;
        for i in 1..=p {
          if t >= i {
            // Standardized residual from step t-i
            let sigma_t_i = (log_sigma2[t - i].exp()).sqrt();
            let z_t_i = x[t - i] / sigma_t_i; // z_{t-i}

            // Add alpha_i(|z_{t-i}| - E|z|) + gamma_i z_{t-i}
            shock_term += self.alpha[i - 1] * (z_t_i.abs() - e_abs_z) + self.gamma[i - 1] * z_t_i;
          }
        }

        // 2) Sum in the log-variance from q lags
        let mut persistence_term = 0.0;
        for j in 1..=q {
          if t >= j {
            persistence_term += self.beta[j - 1] * log_sigma2[t - j];
          }
        }

        // 3) Final log-variance
        log_sigma2[t] = self.omega + shock_term + persistence_term;
      }

      // Convert log_sigma2[t] to sigma_t and compute X_t
      let sigma_t = (log_sigma2[t].exp()).sqrt();
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

#[cfg(test)]
mod tests {
  use ndarray::arr1;

  use crate::{
    plot_1d,
    stochastic::{autoregressive::egarch::EGARCH, SamplingExt},
  };

  #[test]
  fn egarch_plot() {
    let alpha = arr1(&[0.1, 0.05]); // p=2
    let gamma = arr1(&[0.0, -0.02]); // p=2
    let beta = arr1(&[0.8]); // q=1
    let egarchpq = EGARCH::new(0.0, alpha, gamma, beta, 100, None);
    plot_1d!(egarchpq.sample(), "EGARCH(p,q) process");
  }
}
