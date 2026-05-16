//! # Ou
//!
//! $$
//! dX_t=\kappa(\theta-X_t)\,dt+\sigma\,dW_t
//! $$
//!
use ndarray::Array1;
use ndarray::s;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

#[derive(Clone, Copy)]
pub struct Ou<T: FloatExt, S: SeedExt = Unseeded> {
  /// Mean-reversion speed (κ in the SDE `dX = κ(θ − X) dt + σ dW`). Controls
  /// how fast `X` is pulled back toward [`mu`](Self::mu).
  pub theta: T,
  /// Long-run mean level (θ in the SDE). The value `X` reverts to as
  /// `t → ∞`.
  pub mu: T,
  /// Diffusion / noise scale parameter (σ in the SDE).
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Ou<T, S> {
  pub fn new(theta: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    Self {
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Ou<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut ou = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return ou;
    }

    ou[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return ou;
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let drift_scale = self.theta * dt;
    let sqrt_dt = dt.sqrt();
    let diff_scale = self.sigma;
    let mut prev = ou[0];
    let mut tail_view = ou.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("Ou output tail must be contiguous");
    let normal = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &self.seed);
    normal.fill_slice_fast(tail);

    for z in tail.iter_mut() {
      let next = prev + drift_scale * (self.mu - prev) + diff_scale * *z;
      *z = next;
      prev = next;
    }

    ou
  }
}

py_process_1d!(PyOu, Ou,
  sig: (theta, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
