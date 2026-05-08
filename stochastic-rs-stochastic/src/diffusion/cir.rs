//! # Cir
//!
//! $$
//! dX_t=\kappa(\theta-X_t)\,dt+\sigma\sqrt{X_t}\,dW_t
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

/// Cox-Ingersoll-Ross (Cir) process.
///
/// `dX(t) = theta * (mu - X(t)) * dt + sigma * sqrt(X(t)) * dW(t)`
///
/// In the SDE notation `dX = κ(θ − X) dt + σ √X dW` the Rust field
/// [`theta`](Self::theta) corresponds to κ (mean-reversion speed) and
/// [`mu`](Self::mu) corresponds to θ (long-run mean level).
pub struct Cir<T: FloatExt, S: SeedExt = Unseeded> {
  /// Mean-reversion speed (κ in the SDE). Controls how fast `X` is pulled
  /// back toward [`mu`](Self::mu).
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
  /// Enables symmetric/truncated update variant when true.
  pub use_sym: Option<bool>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt> Cir<T> {
  /// Create a new Cir process.
  pub fn new(
    theta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
  ) -> Self {
    assert!(
      T::from_usize_(2) * theta * mu >= sigma.powi(2),
      "2 * theta * mu < sigma^2"
    );

    Self {
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      use_sym,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> Cir<T, Deterministic> {
  pub fn seeded(
    theta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
    seed: u64,
  ) -> Self {
    assert!(
      T::from_usize_(2) * theta * mu >= sigma.powi(2),
      "2 * theta * mu < sigma^2"
    );

    Self {
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      use_sym,
      seed: Deterministic::new(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Cir<T, S> {
  type Output = Array1<T>;

  /// Sample the Cox-Ingersoll-Ross (Cir) process
  fn sample(&self) -> Self::Output {
    let mut cir = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return cir;
    }

    cir[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return cir;
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    let diff_scale = self.sigma;
    let mut prev = cir[0];
    let mut tail_view = cir.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("Cir output tail must be contiguous");
    let normal = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &self.seed);
    normal.fill_slice_fast(tail);

    for z in tail.iter_mut() {
      let dcir = self.theta * (self.mu - prev) * dt + diff_scale * prev.abs().sqrt() * *z;
      let next = match self.use_sym.unwrap_or(false) {
        true => (prev + dcir).abs(),
        false => (prev + dcir).max(T::zero()),
      };
      *z = next;
      prev = next;
    }

    cir
  }
}

py_process_1d!(PyCir, Cir,
  sig: (theta, mu, sigma, n, x0=None, t=None, use_sym=None, seed=None, dtype=None),
  params: (theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, use_sym: Option<bool>)
);
