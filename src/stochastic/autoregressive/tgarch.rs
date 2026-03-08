//! # Tgarch
//!
//! $$
//! \sigma_t^2=\omega+\sum_{i=1}^p(\alpha_i+\gamma_i\mathbf 1_{\{X_{t-i}<0\}})X_{t-i}^2
//! +\sum_{j=1}^q\beta_j\sigma_{t-j}^2
//! $$
//!
use ndarray::Array1;

use crate::distributions::normal::SimdNormal;
use crate::simd_rng::Deterministic;
use crate::simd_rng::Seed;
use crate::simd_rng::Unseeded;
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
pub struct TGARCH<T: FloatExt, S: Seed = Unseeded> {
  /// Constant term in conditional variance dynamics.
  pub omega: T,
  /// Model shape / loading parameter.
  pub alpha: Array1<T>,
  /// Model asymmetry / nonlinearity parameter.
  pub gamma: Array1<T>,
  /// Model slope / loading parameter.
  pub beta: Array1<T>,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt> TGARCH<T> {
  pub fn new(omega: T, alpha: Array1<T>, gamma: Array1<T>, beta: Array1<T>, n: usize) -> Self {
    assert!(omega > T::zero(), "TGARCH requires omega > 0");
    assert!(
      alpha.len() == gamma.len(),
      "TGARCH requires alpha.len() == gamma.len()"
    );
    Self {
      omega,
      alpha,
      gamma,
      beta,
      n,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> TGARCH<T, Deterministic> {
  /// Create a new TGARCH model with a deterministic seed for reproducible output.
  pub fn seeded(
    omega: T,
    alpha: Array1<T>,
    gamma: Array1<T>,
    beta: Array1<T>,
    n: usize,
    seed: u64,
  ) -> Self {
    assert!(omega > T::zero(), "TGARCH requires omega > 0");
    assert!(
      alpha.len() == gamma.len(),
      "TGARCH requires alpha.len() == gamma.len()"
    );
    Self {
      omega,
      alpha,
      gamma,
      beta,
      n,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: Seed> ProcessExt<T> for TGARCH<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let p = self.alpha.len();
    let q = self.beta.len();

    // Standard normal noise
    let mut z = Array1::<T>::zeros(self.n);
    if self.n > 0 {
      let slice = z.as_slice_mut().expect("contiguous");
      let mut seed = self.seed;
      let normal = SimdNormal::<T>::from_seed_source(T::zero(), T::one(), &mut seed);
      normal.fill_slice_fast(slice);
    }

    // Arrays for X_t and sigma_t^2
    let mut x = Array1::<T>::zeros(self.n);
    let mut sigma2 = Array1::<T>::zeros(self.n);
    let var_floor = T::from_f64_fast(1e-12);

    // Sum up alpha + 0.5 gamma + beta for unconditional variance approximation
    let sum_alpha = self.alpha.iter().cloned().sum();
    let sum_gamma_half = self.gamma.iter().cloned().sum::<T>() * T::from_f64_fast(0.5);
    let sum_beta = self.beta.iter().cloned().sum();
    let denom = T::one() - sum_alpha - sum_gamma_half - sum_beta;
    assert!(
      denom > T::zero(),
      "TGARCH requires sum(alpha) + 0.5*sum(gamma) + sum(beta) < 1 for finite unconditional variance"
    );

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
      assert!(
        sigma2[t].is_finite() && sigma2[t] > T::zero(),
        "TGARCH produced non-positive or non-finite conditional variance at t={}",
        t
      );
      // X_t = sigma_t * z_t
      x[t] = sigma2[t].max(var_floor).sqrt() * z[t];
    }

    x
  }
}

py_process_1d!(PyTGARCH, TGARCH,
  sig: (omega, alpha, gamma_, beta, n, dtype=None),
  params: (omega: f64, alpha: Vec<f64>, gamma_: Vec<f64>, beta: Vec<f64>, n: usize)
);
