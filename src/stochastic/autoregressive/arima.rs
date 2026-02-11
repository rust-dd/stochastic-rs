use ndarray::Array1;

use super::ar::ARp;
use super::ma::MAq;
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
  ar: ARp<T>,
}

impl<T: Float> ARIMA<T> {
  /// Create a new ARIMA model with the given parameters.
  pub fn new(ar_coefs: Array1<T>, ma_coefs: Array1<T>, d: usize, sigma: T, n: usize) -> Self {
    let ar = ARp::new(ar_coefs.clone(), sigma, n, None);

    Self {
      ar_coefs,
      ma_coefs,
      d,
      sigma,
      n,
      ar,
    }
  }
}

impl<T: Float> Process<T> for ARIMA<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    // 1) Generate an AR(p) series with user-provided coefficients
    let ar_model = &self.ar;
    let ar_series = ar_model.sample();

    // 2) Generate an MA(q) series with user-provided coefficients
    let ma_model = MAq::new(self.ma_coefs.clone(), self.sigma, self.n);
    let ma_series = ma_model.sample();

    // 3) Summation -> ARMA(p,q)
    let arma_series = &ar_series + &ma_series;

    // 4) Inverse difference d times -> ARIMA(p,d,q)
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

#[cfg(test)]
mod tests {
  use ndarray::arr1;

  use crate::plot_1d;
  use crate::stochastic::autoregressive::arima::ARIMA;
  use crate::stochastic::Process;

  #[test]
  fn arima_plot() {
    // p=2, d=1, q=2
    // AR/MA coefficients (array-based) for demonstration
    let ar_coefs = arr1(&[0.5, -0.1]);
    let ma_coefs = arr1(&[0.2, 0.2]);
    let arima_model = ARIMA::new(
      ar_coefs, ma_coefs, 1,   // d
      1.0, // sigma
      100, // n
    );
    plot_1d!(arima_model.sample(), "ARIMA(p,d,q) process");
  }
}
