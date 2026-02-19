//! # Garch
//!
//! $$
//! \sigma_t^2=\omega+\sum_{i=1}^p\alpha_iX_{t-i}^2+\sum_{j=1}^q\beta_j\sigma_{t-j}^2,\qquad X_t=\sigma_t z_t
//! $$
//!
use ndarray::Array1;

use crate::stochastic::noise::wn::Wn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

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
pub struct GARCH<T: FloatExt> {
  /// Constant term in conditional variance dynamics.
  pub omega: T,
  /// Model shape / loading parameter.
  pub alpha: Array1<T>,
  /// Model slope / loading parameter.
  pub beta: Array1<T>,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  wn: Wn<T>,
}

impl<T: FloatExt> GARCH<T> {
  pub fn new(omega: T, alpha: Array1<T>, beta: Array1<T>, n: usize) -> Self {
    assert!(omega > T::zero(), "GARCH requires omega > 0");
    GARCH {
      omega,
      alpha,
      beta,
      n,
      wn: Wn::new(n, None, None),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for GARCH<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let p = self.alpha.len();
    let q = self.beta.len();

    // Generate white noise z_t
    let z = self.wn.sample();

    // Arrays for X_t and sigma_t^2
    let mut x = Array1::<T>::zeros(self.n);
    let mut sigma2 = Array1::<T>::zeros(self.n);
    let var_floor = T::from_f64_fast(1e-12);

    // Sum of alpha/beta for unconditional variance initialization
    let sum_alpha = self.alpha.iter().cloned().sum();
    let sum_beta = self.beta.iter().cloned().sum();
    let denom = T::one() - sum_alpha - sum_beta;
    assert!(
      denom > T::zero(),
      "GARCH requires sum(alpha) + sum(beta) < 1 for finite unconditional variance"
    );

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
      assert!(
        sigma2[t].is_finite() && sigma2[t] > T::zero(),
        "GARCH produced non-positive or non-finite conditional variance at t={}",
        t
      );
      // X_t = sigma_t * z[t]
      x[t] = sigma2[t].max(var_floor).sqrt() * z[t];
    }

    x
  }
}

py_process_1d!(PyGARCH, GARCH,
  sig: (omega, alpha, beta, n, dtype=None),
  params: (omega: f64, alpha: Vec<f64>, beta: Vec<f64>, n: usize)
);
