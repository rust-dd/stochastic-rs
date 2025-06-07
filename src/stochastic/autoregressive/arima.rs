use crate::stochastic::SamplingExt;
use impl_new_derive::ImplNew;
use ndarray::Array1;

use super::{ar::ARp, ma::MAq};

/// Implements an ARIMA(p, d, q) process using explicit backshift notation:
///
/// \[
///   \phi(B)\,(1 - B)^d X_t = \theta(B)\,\epsilon_t,
/// \]
/// where \(\phi(B)\) and \(\theta(B)\) are polynomials of orders p and q, respectively,
/// and \(B\) is the backshift (lag) operator (\(B X_t = X_{t-1}\)).
#[derive(ImplNew)]
pub struct ARIMA {
  /// AR coefficients (\(\phi_1,\dots,\phi_p\)) as an Array1
  pub ar_coefs: Array1<f64>,
  /// MA coefficients (\(\theta_1,\dots,\theta_q\)) as an Array1
  pub ma_coefs: Array1<f64>,
  /// Differencing order (d)
  pub d: usize,
  /// Noise std dev (\(\sigma\)) for the innovations
  pub sigma: f64,
  /// Final length of time series
  pub n: usize,
  /// Optional batch size
  pub m: Option<usize>,
}

impl SamplingExt<f64> for ARIMA {
  fn sample(&self) -> Array1<f64> {
    // 1) Generate an AR(p) series with user-provided coefficients
    let ar_model = ARp::new(
      self.ar_coefs.clone(),
      self.sigma,
      self.n,
      None, // batch
      None, // x0
    );
    let ar_series = ar_model.sample();

    // 2) Generate an MA(q) series with user-provided coefficients
    let ma_model = MAq::new(self.ma_coefs.clone(), self.sigma, self.n, None);
    let ma_series = ma_model.sample();

    // 3) Summation -> ARMA(p,q)
    let arma_series = &ar_series + &ma_series;

    // 4) Inverse difference d times -> ARIMA(p,d,q)
    let mut result = arma_series;
    for _ in 0..self.d {
      result = inverse_difference(&result);
    }

    result
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

/// Inverse differencing once, converting Y into X:
/// X[0] = Y[0],  X[t] = X[t-1] + Y[t], for t=1..(n-1).
fn inverse_difference(y: &Array1<f64>) -> Array1<f64> {
  let n = y.len();
  if n == 0 {
    return y.clone();
  }
  let mut x = Array1::<f64>::zeros(n);
  x[0] = y[0];
  for t in 1..n {
    x[t] = x[t - 1] + y[t];
  }
  x
}

#[cfg(test)]
mod tests {
  use ndarray::arr1;

  use crate::{
    plot_1d,
    stochastic::{autoregressive::arima::ARIMA, SamplingExt},
  };

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
      None,
    );
    plot_1d!(arima_model.sample(), "ARIMA(p,d,q) process");
  }
}
