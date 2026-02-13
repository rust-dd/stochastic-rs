use ndarray::Array1;

use crate::stochastic::noise::wn::Wn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

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
///   Must have the same length as `alpha`.
/// - `beta`:  Array \(\{\beta_1, \ldots, \beta_q\}\) for the past variance terms.
/// - `n`:     Length of the time series to generate.
/// - `m`:     Optional batch size (unused by default).
///
/// # Notes
/// - Stationarity constraints typically include: \(\sum \alpha_i + \tfrac{1}{2}\sum \gamma_i + \sum \beta_j < 1\).
/// - We do a simple unconditional variance initialization for \(\sigma_0^2\).
pub struct TGARCH<T: FloatExt> {
  pub omega: T,
  pub alpha: Array1<T>,
  pub gamma: Array1<T>,
  pub beta: Array1<T>,
  pub n: usize,
  wn: Wn<T>,
}

impl<T: FloatExt> TGARCH<T> {
  pub fn new(omega: T, alpha: Array1<T>, gamma: Array1<T>, beta: Array1<T>, n: usize) -> Self {
    Self {
      omega,
      alpha,
      gamma,
      beta,
      n,
      wn: Wn::new(n, None, None),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for TGARCH<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let p = self.alpha.len();
    let q = self.beta.len();

    // Standard normal noise
    let z = self.wn.sample();

    // Arrays for X_t and sigma_t^2
    let mut x = Array1::<T>::zeros(self.n);
    let mut sigma2 = Array1::<T>::zeros(self.n);

    // Sum up alpha + 0.5 gamma + beta for unconditional variance approximation
    let sum_alpha = self.alpha.iter().cloned().sum();
    let sum_gamma_half = self.gamma.iter().cloned().sum::<T>() * T::from_f64_fast(0.5);
    let sum_beta = self.beta.iter().cloned().sum();
    let denom = (T::one() - sum_alpha - sum_gamma_half - sum_beta).max(T::from_f64_fast(1e-8));

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
            let indicator = if x_lag < T::zero() {
              T::one()
            } else {
              T::zero()
            };

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
}
