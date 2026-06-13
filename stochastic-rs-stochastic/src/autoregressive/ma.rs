//! # Ma
//!
//! $$
//! X_t=\varepsilon_t+\sum_{k=1}^q\theta_k\varepsilon_{t-k},\qquad \varepsilon_t\sim\mathcal N(0,\sigma^2)
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

impl<T: FloatExt, S: SeedExt> MAq<T, S> {
  /// Create a new MA(q) model with the given parameters.
  pub fn new(theta: Array1<T>, sigma: T, n: usize, seed: S) -> Self {
    assert!(sigma > T::zero(), "MAq requires sigma > 0");
    Self {
      theta,
      sigma,
      n,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for MAq<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = MAqSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> MAqSampler<T> {
    MAqSampler {
      n: self.n,
      theta: self.theta.clone(),
      normal: SimdNormal::<T>::new(T::zero(), self.sigma, &self.seed),
    }
  }
}

/// Reusable [`MAq`] sampling state: owns the Gaussian innovation source and the
/// MA coefficients so a Monte-Carlo loop pays the `SimdNormal` setup once.
#[doc(hidden)]
pub struct MAqSampler<T: FloatExt> {
  n: usize,
  theta: Array1<T>,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> MAqSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    let n = out.len();
    let q = self.theta.len();

    let mut noise = Array1::<T>::zeros(n);
    if n > 0 {
      let slice = noise.as_slice_mut().expect("contiguous");
      self.normal.fill_slice_fast(slice);
    }

    // MA recursion
    for t in 0..n {
      // Start with current noise
      let mut val = noise[t];
      // Add in past noises scaled by theta
      for k in 1..=q {
        if t >= k {
          val += self.theta[k - 1] * noise[t - k];
        }
      }
      out[t] = val;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for MAqSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("MAq output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyMAq, MAq,
  sig: (theta, sigma, n, seed=None, dtype=None),
  params: (theta: Vec<f64>, sigma: f64, n: usize)
);
