//! # Feller
//!
//! $$
//! dX_t=a(t,X_t)dt+b(t,X_t)dW_t
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

/// Feller–logistic diffusion
/// dX_t = kappa (theta - X_t) X_t dt + sigma sqrt(X_t) dW_t
pub struct FellerLogistic<T: FloatExt, S: SeedExt = Unseeded> {
  /// Mean-reversion speed parameter.
  pub kappa: T,
  /// Long-run target level / model location parameter.
  pub theta: T,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// If true, reflect at 0; otherwise clamp at 0
  pub use_sym: Option<bool>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt> FellerLogistic<T> {
  pub fn new(
    kappa: T,
    theta: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
  ) -> Self {
    Self {
      kappa,
      theta,
      sigma,
      n,
      x0,
      t,
      use_sym,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> FellerLogistic<T, Deterministic> {
  pub fn seeded(
    kappa: T,
    theta: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
    seed: u64,
  ) -> Self {
    Self {
      kappa,
      theta,
      sigma,
      n,
      x0,
      t,
      use_sym,
      seed: Deterministic::new(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for FellerLogistic<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut x = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return x;
    }

    x[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return x;
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    let diff_scale = self.sigma;
    let mut prev = x[0];
    let mut tail_view = x.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("Feller output tail must be contiguous");
    let normal = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &self.seed);
    normal.fill_slice_fast(tail);

    for z in tail.iter_mut() {
      let xi = prev.max(T::zero());
      let drift = self.kappa * (self.theta - xi) * xi * dt;
      let diff = diff_scale * xi.sqrt() * *z;
      let next = xi + drift + diff;
      let clamped = match self.use_sym.unwrap_or(false) {
        true => next.abs(),
        false => next.max(T::zero()),
      };
      *z = clamped;
      prev = clamped;
    }

    x
  }
}

py_process_1d!(PyFellerLogistic, FellerLogistic,
  sig: (kappa, theta, sigma, n, x0=None, t=None, use_sym=None, seed=None, dtype=None),
  params: (kappa: f64, theta: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, use_sym: Option<bool>)
);
