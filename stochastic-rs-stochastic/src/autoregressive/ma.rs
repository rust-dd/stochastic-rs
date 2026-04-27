//! # Ma
//!
//! $$
//! X_t=\varepsilon_t+\sum_{k=1}^q\theta_k\varepsilon_{t-k},\qquad \varepsilon_t\sim\mathcal N(0,\sigma^2)
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Implements an MA(q) model:
///
/// \[
///   X_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \dots + \theta_q \epsilon_{t-q},
///   \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2).
/// \]
///
/// # Fields
/// - `theta`: MA coefficients (\(\theta_1, \ldots, \theta_q\)).
/// - `sigma`: Standard deviation of noise \(\epsilon_t\).
/// - `n`: Length of time series.
/// - `m`: Optional batch size.
#[derive(Debug, Clone)]
pub struct MAq<T: FloatExt, S: SeedExt = Unseeded> {
  /// MA coefficients
  pub theta: Array1<T>,
  /// Noise std dev
  pub sigma: T,
  /// Number of observations
  pub n: usize,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt> MAq<T> {
  /// Create a new MA(q) model with the given parameters.
  pub fn new(theta: Array1<T>, sigma: T, n: usize) -> Self {
    assert!(sigma > T::zero(), "MAq requires sigma > 0");
    Self {
      theta,
      sigma,
      n,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> MAq<T, Deterministic> {
  /// Create a new MA(q) model with a deterministic seed for reproducible output.
  pub fn seeded(theta: Array1<T>, sigma: T, n: usize, seed: u64) -> Self {
    assert!(sigma > T::zero(), "MAq requires sigma > 0");
    Self {
      theta,
      sigma,
      n,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for MAq<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let q = self.theta.len();
    let mut noise = Array1::<T>::zeros(self.n);
    if self.n > 0 {
      let slice = noise.as_slice_mut().expect("contiguous");
      let mut seed = self.seed;
      let normal = SimdNormal::<T>::from_seed_source(T::zero(), self.sigma, &mut seed);
      normal.fill_slice_fast(slice);
    }
    let mut series = Array1::<T>::zeros(self.n);

    // MA recursion
    for t in 0..self.n {
      // Start with current noise
      let mut val = noise[t];
      // Add in past noises scaled by theta
      for k in 1..=q {
        if t >= k {
          val += self.theta[k - 1] * noise[t - k];
        }
      }
      series[t] = val;
    }

    series
  }
}

py_process_1d!(PyMAq, MAq,
  sig: (theta, sigma, n, seed=None, dtype=None),
  params: (theta: Vec<f64>, sigma: f64, n: usize)
);
