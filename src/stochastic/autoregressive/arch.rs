use ndarray::Array1;

use crate::stochastic::noise::wn::Wn;
use crate::stochastic::Float;
use crate::stochastic::Process;

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
pub struct ARCH<T: Float> {
  /// Omega (constant term in variance)
  pub omega: T,
  /// Coefficients alpha_i
  pub alpha: Array1<T>,
  /// Length of series
  pub n: usize,
  wn: Wn<T>,
}

impl<T: Float> ARCH<T> {
  /// Create a new ARCH model.
  pub fn new(omega: T, alpha: Array1<T>, n: usize) -> Self {
    Self {
      omega,
      alpha,
      n,
      wn: Wn::new(n, None, None),
    }
  }
}

impl<T: Float> Process<T> for ARCH<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let m = self.alpha.len();
    let z = self.wn.sample();
    let mut x = Array1::<T>::zeros(self.n);

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
}

#[cfg(test)]
mod tests {
  use ndarray::arr1;

  use crate::plot_1d;
  use crate::stochastic::autoregressive::arch::ARCH;
  use crate::stochastic::Process;

  #[test]
  fn arch_plot() {
    let alpha = arr1(&[0.2, 0.1]);
    let arch_model = ARCH::new(0.1, alpha, 100);
    plot_1d!(arch_model.sample(), "ARCH(m) process");
  }
}
