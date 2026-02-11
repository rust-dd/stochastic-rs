use ndarray::Array1;

use crate::stochastic::noise::wn::Wn;
use crate::stochastic::Float;
use crate::stochastic::Process;

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
pub struct AGARCH<T: Float> {
  pub omega: T,
  pub alpha: Array1<T>,
  pub delta: Array1<T>,
  pub beta: Array1<T>,
  pub n: usize,
  wn: Wn<T>,
}

impl<T: Float> AGARCH<T> {
  pub fn new(omega: T, alpha: Array1<T>, delta: Array1<T>, beta: Array1<T>, n: usize) -> Self {
    Self {
      omega,
      alpha,
      delta,
      beta,
      n,
      wn: Wn::new(n, None, None),
    }
  }
}

impl<T: Float> Process<T> for AGARCH<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let p = self.alpha.len();
    let q = self.beta.len();

    // Generate white noise
    let z = &self.wn.sample();

    // Arrays for X_t and sigma_t^2
    let mut x = Array1::<T>::zeros(self.n);
    let mut sigma2 = Array1::<T>::zeros(self.n);

    // Summation for unconditional variance init
    let sum_alpha = self.alpha.iter().cloned().sum();
    let sum_delta_half = self.delta.iter().cloned().sum::<T>() + T::from_f64_fast(0.5);
    let sum_beta = self.beta.iter().cloned().sum();
    let denom = (T::one() - sum_alpha - sum_delta_half - sum_beta).max(T::from_f64_fast(1e-8));

    for t in 0..self.n {
      if t == 0 {
        sigma2[t] = self.omega / denom;
      } else {
        let mut var_t = self.omega;
        // p-lag terms
        for i in 1..=p {
          if t >= i {
            let x_lag = x[t - i];
            let indicator = if x_lag < T::zero() {
              T::one()
            } else {
              T::zero()
            };

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
}

#[cfg(test)]
mod tests {
  use ndarray::arr1;

  use crate::plot_1d;
  use crate::stochastic::autoregressive::agrach::AGARCH;
  use crate::stochastic::Process;

  #[test]
  fn agarch_plot() {
    let alpha = arr1(&[0.05, 0.01]); // p=2
    let delta = arr1(&[0.03, 0.01]); // p=2
    let beta = arr1(&[0.8]); // q=1
    let agarchpq = AGARCH::new(0.1, alpha, delta, beta, 100);
    plot_1d!(agarchpq.sample(), "A-GARCH(p,q) process");
  }
}
