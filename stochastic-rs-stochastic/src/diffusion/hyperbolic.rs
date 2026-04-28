//! # Hyperbolic
//!
//! $$
//! dX_t=-\kappa\frac{X_t}{\sqrt{1+X_t^2}}\,dt+\sigma\,dW_t
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

pub struct Hyperbolic<T: FloatExt, S: SeedExt = Unseeded> {
  /// Mean-reversion speed parameter.
  pub kappa: T,
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

impl<T: FloatExt> Hyperbolic<T> {
  pub fn new(kappa: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      kappa,
      sigma,
      n,
      x0,
      t,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> Hyperbolic<T, Deterministic> {
  pub fn seeded(kappa: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: u64) -> Self {
    Self {
      kappa,
      sigma,
      n,
      x0,
      t,
      seed: Deterministic::new(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Hyperbolic<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut hyperbolic = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return hyperbolic;
    }

    hyperbolic[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return hyperbolic;
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    let mut prev = hyperbolic[0];
    let mut tail_view = hyperbolic.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("Hyperbolic output tail must be contiguous");
    let normal = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &self.seed);
    normal.fill_slice_fast(tail);

    for z in tail.iter_mut() {
      let next =
        prev + (-self.kappa * prev / (T::one() + prev * prev).sqrt()) * dt + self.sigma * *z;
      *z = next;
      prev = next;
    }

    hyperbolic
  }
}

py_process_1d!(PyHyperbolic, Hyperbolic,
  sig: (kappa, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (kappa: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
