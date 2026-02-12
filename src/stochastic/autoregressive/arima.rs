use ndarray::Array1;

use crate::stochastic::noise::wn::Wn;
use crate::stochastic::Float;
use crate::stochastic::Process;

/// Implements an ARIMA(p, d, q) process using explicit backshift notation:
///
/// \[
///   \phi(B)\,(1 - B)^d X_t = \theta(B)\,\epsilon_t,
/// \]
/// where \(\phi(B)\) and \(\theta(B)\) are polynomials of orders p and q, respectively,
/// and \(B\) is the backshift (lag) operator (\(B X_t = X_{t-1}\)).
pub struct ARIMA<T: Float> {
  /// AR coefficients (\(\phi_1,\dots,\phi_p\)) as an Array1
  pub ar_coefs: Array1<T>,
  /// MA coefficients (\(\theta_1,\dots,\theta_q\)) as an Array1
  pub ma_coefs: Array1<T>,
  /// Differencing order (d)
  pub d: usize,
  /// Noise std dev (\(\sigma\)) for the innovations
  pub sigma: T,
  /// Final length of time series
  pub n: usize,
  wn: Wn<T>,
}

impl<T: Float> ARIMA<T> {
  /// Create a new ARIMA model with the given parameters.
  pub fn new(ar_coefs: Array1<T>, ma_coefs: Array1<T>, d: usize, sigma: T, n: usize) -> Self {
    Self {
      ar_coefs,
      ma_coefs,
      d,
      sigma,
      n,
      wn: Wn::new(n, None, Some(sigma)),
    }
  }
}

impl<T: Float> Process<T> for ARIMA<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let p = self.ar_coefs.len();
    let q = self.ma_coefs.len();
    let noise = self.wn.sample();
    let mut arma_series = Array1::<T>::zeros(self.n);

    // Single-pass ARMA(p,q) recursion with shared noise:
    // X_t = sum_k(phi_k * X_{t-k}) + eps_t + sum_k(theta_k * eps_{t-k})
    for t in 0..self.n {
      let mut val = noise[t];

      for k in 1..=p {
        if t >= k {
          val += self.ar_coefs[k - 1] * arma_series[t - k];
        }
      }

      for k in 1..=q {
        if t >= k {
          val += self.ma_coefs[k - 1] * noise[t - k];
        }
      }

      arma_series[t] = val;
    }

    // Inverse difference d times -> ARIMA(p,d,q)
    let mut result = arma_series;
    for _ in 0..self.d {
      result = Self::inverse_difference(&result);
    }

    result
  }
}

impl<T: Float> ARIMA<T> {
  /// Inverse differencing once, converting Y into X:
  /// X[0] = Y[0],  X[t] = X[t-1] + Y[t], for t=1..(n-1).
  fn inverse_difference(y: &Array1<T>) -> Array1<T> {
    let n = y.len();
    if n == 0 {
      return y.clone();
    }
    let mut x = Array1::<T>::zeros(n);
    x[0] = y[0];
    for t in 1..n {
      x[t] = x[t - 1] + y[t];
    }
    x
  }
}
