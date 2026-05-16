//! # Ar
//!
//! $$
//! X_t=\sum_{k=1}^p \phi_k X_{t-k}+\varepsilon_t,\qquad \varepsilon_t\sim\mathcal N(0,\sigma^2)
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Implements an AR(p) model:
///
/// \[
///   X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \dots + \phi_p X_{t-p}
///         + \epsilon_t,
///   \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2).
/// \]
///
/// # Fields
/// - `phi`: Vector of AR coefficients (\(\phi_1, \ldots, \phi_p\)).
/// - `sigma`: Standard deviation of the noise \(\epsilon_t\).
/// - `n`: Length of the time series.
/// - `m`: Optional batch size (for parallel sampling).
/// - `x0`: Optional array of initial values. If provided, should have length at least `phi.len()`.
#[derive(Debug, Clone)]
pub struct ARp<T: FloatExt, S: SeedExt = Unseeded> {
  /// AR coefficients
  pub phi: Array1<T>,
  /// Noise std dev
  pub sigma: T,
  /// Number of observations
  pub n: usize,
  /// Optional initial conditions
  pub x0: Option<Array1<T>>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> ARp<T, S> {
  /// Create a new AR process with given coefficients and noise standard deviation.
  pub fn new(phi: Array1<T>, sigma: T, n: usize, x0: Option<Array1<T>>, seed: S) -> Self {
    assert!(sigma > T::zero(), "ARp requires sigma > 0");
    if let Some(init) = &x0 {
      let required = phi.len().min(n);
      assert!(
        init.len() >= required,
        "ARp requires x0.len() >= min(phi.len(), n) (got {}, need at least {})",
        init.len(),
        required
      );
    }
    Self {
      phi,
      sigma,
      n,
      x0,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for ARp<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let p = self.phi.len();
    let mut noise = Array1::<T>::zeros(self.n);
    if self.n > 0 {
      let slice = noise.as_slice_mut().expect("contiguous");
      let normal = SimdNormal::<T>::from_seed_source(T::zero(), self.sigma, &self.seed);
      normal.fill_slice_fast(slice);
    }
    let noise = &noise;
    let mut series = Array1::<T>::zeros(self.n);

    // Fill initial conditions if provided
    if let Some(init) = &self.x0 {
      // Copy up to min(p, n)
      for i in 0..p.min(self.n) {
        series[i] = init[i];
      }
    }

    // AR recursion
    let start = if self.x0.is_some() { p.min(self.n) } else { 0 };
    for t in start..self.n {
      let mut val = T::zero();
      for k in 1..=p {
        if t >= k {
          val += self.phi[k - 1] * series[t - k];
        }
      }
      series[t] = val + noise[t];
    }

    series
  }
}

py_process_1d!(PyARp, ARp,
  sig: (phi, sigma, n, x0=None, seed=None, dtype=None),
  params: (phi: Vec<f64>, sigma: f64, n: usize, x0: Option<Vec<f64>>)
);
