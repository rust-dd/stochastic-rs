//! # Arima
//!
//! $$
//! \phi(B)(1-B)^dX_t=\theta(B)\varepsilon_t,\qquad \varepsilon_t\sim\mathcal N(0,\sigma^2)
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Implements an Arima(p, d, q) process using explicit backshift notation:
///
/// \[
///   \phi(B)\,(1 - B)^d X_t = \theta(B)\,\epsilon_t,
/// \]
/// where \(\phi(B)\) and \(\theta(B)\) are polynomials of orders p and q, respectively,
/// and \(B\) is the backshift (lag) operator (\(B X_t = X_{t-1}\)).
#[derive(Debug, Clone)]
pub struct Arima<T: FloatExt, S: SeedExt = Unseeded> {
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
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Arima<T, S> {
  /// Create a new Arima model with the given parameters.
  pub fn new(
    ar_coefs: Array1<T>,
    ma_coefs: Array1<T>,
    d: usize,
    sigma: T,
    n: usize,
    seed: S,
  ) -> Self {
    assert!(sigma > T::zero(), "Arima requires sigma > 0");
    Self {
      ar_coefs,
      ma_coefs,
      d,
      sigma,
      n,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Arima<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let p = self.ar_coefs.len();
    let q = self.ma_coefs.len();
    let mut noise = Array1::<T>::zeros(self.n);
    if self.n > 0 {
      let slice = noise.as_slice_mut().expect("contiguous");
      let normal = SimdNormal::<T>::new(T::zero(), self.sigma, &self.seed);
      normal.fill_slice_fast(slice);
    }
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

    // Inverse difference d times -> Arima(p,d,q)
    let mut result = arma_series;
    for _ in 0..self.d {
      result = Self::inverse_difference(&result);
    }

    result
  }
}

impl<T: FloatExt, S: SeedExt> Arima<T, S> {
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

py_process_1d!(PyArima, Arima,
  sig: (ar_coefs, ma_coefs, d, sigma, n, seed=None, dtype=None),
  params: (ar_coefs: Vec<f64>, ma_coefs: Vec<f64>, d: usize, sigma: f64, n: usize)
);
