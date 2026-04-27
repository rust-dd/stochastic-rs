//! # ThreeHalf
//!
//! $$
//! dX_t=\kappa X_t(\mu-X_t)\,dt+\sigma X_t^{3/2}\,dW_t
//! $$
//!
use ndarray::Array1;
use ndarray::s;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct ThreeHalf<T: FloatExt, S: SeedExt = Unseeded> {
  /// Mean-reversion speed parameter.
  pub kappa: T,
  /// Long-run mean-level parameter.
  pub mu: T,
  /// Diffusion / noise scale parameter.
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

impl<T: FloatExt> ThreeHalf<T> {
  pub fn new(kappa: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      kappa,
      mu,
      sigma,
      n,
      x0,
      t,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> ThreeHalf<T, Deterministic> {
  pub fn seeded(
    kappa: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: u64,
  ) -> Self {
    Self {
      kappa,
      mu,
      sigma,
      n,
      x0,
      t,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for ThreeHalf<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut three_half = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return three_half;
    }

    three_half[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return three_half;
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    let mut prev = three_half[0];
    let mut tail_view = three_half.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("ThreeHalf output tail must be contiguous");
    let mut seed = self.seed;
    let normal = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &mut seed);
    normal.fill_slice_fast(tail);

    for z in tail.iter_mut() {
      let next = prev
        + self.kappa * prev * (self.mu - prev) * dt
        + self.sigma * prev.abs().powf(T::from_f64_fast(1.5)) * *z;
      *z = next;
      prev = next;
    }

    three_half
  }
}

py_process_1d!(PyThreeHalf, ThreeHalf,
  sig: (kappa, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (kappa: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
