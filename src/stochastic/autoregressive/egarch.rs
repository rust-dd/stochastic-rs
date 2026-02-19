//! # Egarch
//!
//! $$
//! \log(\sigma_t^2)=\omega+\sum_{i=1}^p[\alpha_i(|z_{t-i}|-\mathbb E|z|)+\gamma_i z_{t-i}]
//! +\sum_{j=1}^q\beta_j\log(\sigma_{t-j}^2)
//! $$
//!
use ndarray::Array1;

use crate::stochastic::noise::wn::Wn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

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
///   Must be the same length as `alpha`.
/// - `beta`:  An array \(\{\beta_1, \dots, \beta_q\}\) controlling persistence of past log-variance.
/// - `n`: The number of observations to generate.
/// - `m`: Optional batch size for parallel sampling (unused by default).
///
/// # Notes
/// 1. We assume that `alpha` and `gamma` each have length \(p\).
/// 2. We assume that `beta` has length \(q\).
/// 3. Real-world usage typically enforces constraints to ensure stationarity/ergodicity.
pub struct EGARCH<T: FloatExt> {
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
  wn: Wn<T>,
}

impl<T: FloatExt> EGARCH<T> {
  /// Create a new EGARCH model with the given parameters.
  pub fn new(omega: T, alpha: Array1<T>, gamma: Array1<T>, beta: Array1<T>, n: usize) -> Self {
    assert!(
      alpha.len() == gamma.len(),
      "EGARCH requires alpha.len() == gamma.len()"
    );
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

impl<T: FloatExt> ProcessExt<T> for EGARCH<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let p = self.alpha.len();
    let q = self.beta.len();

    // Generate white noise z_t ~ N(0,1)
    let z = self.wn.sample();

    // Allocate arrays for the time series (X_t) and log of variance (log_sigma2)
    let mut x = Array1::<T>::zeros(self.n);
    let mut log_sigma2 = Array1::<T>::zeros(self.n);

    // For normal(0,1), the expected absolute value is sqrt(2/pi)
    let e_abs_z = (2.0 / std::f64::consts::PI).sqrt();

    for t in 0..self.n {
      if t == 0 {
        // Initialize log-variance (e.g., with omega)
        log_sigma2[t] = self.omega;
      } else {
        // 1) Compute the shock term from p lags
        let mut shock_term = T::zero();
        for i in 1..=p {
          if t >= i {
            // Standardized residual from step t-i
            let sigma_t_i = (log_sigma2[t - i].exp()).sqrt();
            let z_t_i = x[t - i] / sigma_t_i; // z_{t-i}

            // Add alpha_i(|z_{t-i}| - E|z|) + gamma_i z_{t-i}
            shock_term += self.alpha[i - 1] * (z_t_i.abs() - T::from_f64_fast(e_abs_z))
              + self.gamma[i - 1] * z_t_i;
          }
        }

        // 2) Sum in the log-variance from q lags
        let mut persistence_term = T::zero();
        for j in 1..=q {
          if t >= j {
            persistence_term += self.beta[j - 1] * log_sigma2[t - j];
          }
        }

        // 3) Final log-variance
        log_sigma2[t] = self.omega + shock_term + persistence_term;
      }

      // Convert log_sigma2[t] to sigma_t and compute X_t
      assert!(
        log_sigma2[t].is_finite(),
        "EGARCH produced non-finite log-variance at t={}",
        t
      );
      let sigma_t = (log_sigma2[t].exp()).sqrt();
      assert!(
        sigma_t.is_finite() && sigma_t > T::zero(),
        "EGARCH produced non-positive or non-finite sigma at t={}",
        t
      );
      x[t] = sigma_t * z[t];
    }

    x
  }
}

py_process_1d!(PyEGARCH, EGARCH,
  sig: (omega, alpha, gamma_, beta, n, dtype=None),
  params: (omega: f64, alpha: Vec<f64>, gamma_: Vec<f64>, beta: Vec<f64>, n: usize)
);
