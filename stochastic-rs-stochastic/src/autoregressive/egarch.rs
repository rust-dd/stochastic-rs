//! # Egarch
//!
//! $$
//! \log(\sigma_t^2)=\omega+\sum_{i=1}^p[\alpha_i(|z_{t-i}|-\mathbb E|z|)+\gamma_i z_{t-i}]
//! +\sum_{j=1}^q\beta_j\log(\sigma_{t-j}^2)
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Implements an Egarch(p,q) model:
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
#[derive(Debug, Clone)]
pub struct Egarch<T: FloatExt, S: SeedExt = Unseeded> {
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
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Egarch<T, S> {
  /// Create a new Egarch model with the given parameters.
  pub fn new(
    omega: T,
    alpha: Array1<T>,
    gamma: Array1<T>,
    beta: Array1<T>,
    n: usize,
    seed: S,
  ) -> Self {
    assert!(
      alpha.len() == gamma.len(),
      "Egarch requires alpha.len() == gamma.len()"
    );
    Self {
      omega,
      alpha,
      gamma,
      beta,
      n,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Egarch<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = EgarchSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> EgarchSampler<T> {
    EgarchSampler {
      n: self.n,
      omega: self.omega,
      alpha: self.alpha.clone(),
      gamma: self.gamma.clone(),
      beta: self.beta.clone(),
      normal: SimdNormal::<T>::new(T::zero(), T::one(), &self.seed),
    }
  }
}

/// Reusable [`Egarch`] sampling state: owns the standard-normal innovation
/// source and the log-variance coefficients so a Monte-Carlo loop pays the
/// `SimdNormal` setup once.
#[doc(hidden)]
pub struct EgarchSampler<T: FloatExt> {
  n: usize,
  omega: T,
  alpha: Array1<T>,
  gamma: Array1<T>,
  beta: Array1<T>,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> EgarchSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    let n = out.len();
    let p = self.alpha.len();
    let q = self.beta.len();

    // Generate white noise z_t ~ N(0,1)
    let mut z = Array1::<T>::zeros(n);
    if n > 0 {
      let slice = z.as_slice_mut().expect("contiguous");
      self.normal.fill_slice_fast(slice);
    }

    // Scratch array for log of variance (the output buffer holds X_t)
    let mut log_sigma2 = Array1::<T>::zeros(n);

    // For normal(0,1), the expected absolute value is sqrt(2/pi)
    let e_abs_z = T::from_f64_fast((2.0_f64 / std::f64::consts::PI).sqrt());

    for t in 0..n {
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
            let z_t_i = out[t - i] / sigma_t_i; // z_{t-i}

            // Add alpha_i(|z_{t-i}| - E|z|) + gamma_i z_{t-i}
            shock_term += self.alpha[i - 1] * (z_t_i.abs() - e_abs_z) + self.gamma[i - 1] * z_t_i;
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
        "Egarch produced non-finite log-variance at t={}",
        t
      );
      let sigma_t = (log_sigma2[t].exp()).sqrt();
      assert!(
        sigma_t.is_finite() && sigma_t > T::zero(),
        "Egarch produced non-positive or non-finite sigma at t={}",
        t
      );
      out[t] = sigma_t * z[t];
    }
  }
}

impl<T: FloatExt> PathSampler<T> for EgarchSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("Egarch output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyEgarch, Egarch,
  sig: (omega, alpha, gamma_, beta, n, seed=None, dtype=None),
  params: (omega: f64, alpha: Vec<f64>, gamma_: Vec<f64>, beta: Vec<f64>, n: usize)
);
