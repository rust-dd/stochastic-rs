//! # Feller
//!
//! $$
//! dX_t=a(t,X_t)dt+b(t,X_t)dW_t
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

impl<T: FloatExt, S: SeedExt> FellerLogistic<T, S> {
  pub fn new(
    kappa: T,
    theta: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
    seed: S,
  ) -> Self {
    Self {
      kappa,
      theta,
      sigma,
      n,
      x0,
      t,
      use_sym,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for FellerLogistic<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = FellerLogisticSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> FellerLogisticSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    FellerLogisticSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      kappa: self.kappa,
      theta: self.theta,
      diff_scale: self.sigma,
      use_sym: self.use_sym.unwrap_or(false),
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`FellerLogistic`] sampling state.
#[doc(hidden)]
pub struct FellerLogisticSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  kappa: T,
  theta: T,
  diff_scale: T,
  use_sym: bool,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> FellerLogisticSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.normal.fill_slice_fast(tail);
    let mut prev = self.x0;
    for z in tail.iter_mut() {
      let xi = prev.max(T::zero());
      let drift = self.kappa * (self.theta - xi) * xi * self.dt;
      let diff = self.diff_scale * xi.sqrt() * *z;
      let next = xi + drift + diff;
      let clamped = match self.use_sym {
        true => next.abs(),
        false => next.max(T::zero()),
      };
      *z = clamped;
      prev = clamped;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for FellerLogisticSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("Feller output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyFellerLogistic, FellerLogistic,
  sig: (kappa, theta, sigma, n, x0=None, t=None, use_sym=None, seed=None, dtype=None),
  params: (kappa: f64, theta: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, use_sym: Option<bool>)
);
