use ndarray::Array1;

use crate::stochastic::noise::wn::Wn;
use crate::stochastic::Float;
use crate::stochastic::ProcessExt;

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
pub struct GARCH<T: Float> {
  pub omega: T,
  pub alpha: Array1<T>,
  pub beta: Array1<T>,
  pub n: usize,
  wn: Wn<T>,
}

impl<T: Float> GARCH<T> {
  pub fn new(omega: T, alpha: Array1<T>, beta: Array1<T>, n: usize) -> Self {
    GARCH {
      omega,
      alpha,
      beta,
      n,
      wn: Wn::new(n, None, None),
    }
  }
}

impl<T: Float> ProcessExt<T> for GARCH<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let p = self.alpha.len();
    let q = self.beta.len();

    // Generate white noise z_t
    let z = self.wn.sample();

    // Arrays for X_t and sigma_t^2
    let mut x = Array1::<T>::zeros(self.n);
    let mut sigma2 = Array1::<T>::zeros(self.n);

    // Sum of alpha/beta for unconditional variance initialization
    let sum_alpha = self.alpha.iter().cloned().sum();
    let sum_beta = self.beta.iter().cloned().sum();
    let denom = (T::one() - sum_alpha - sum_beta).max(T::from_f64_fast(1e-8));

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
}
