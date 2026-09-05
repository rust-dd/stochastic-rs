use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::uniform::SimdUniform;

use super::sample_positive_stable;
use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Alpha-stable subordinator with Laplace exponent `phi(lambda) = c * lambda^alpha`.
pub struct AlphaStableSubordinator<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Stability index in `(0, 1)`.
  pub alpha: T,
  /// Laplace scale coefficient `c > 0`.
  pub c: T,
  /// Number of grid points.
  pub n: usize,
  /// Initial level.
  pub x0: Option<T>,
  /// Horizon `T`; defaults to `1`.
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> AlphaStableSubordinator<T, S> {
  pub fn new(alpha: T, c: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    assert!(
      alpha > T::zero() && alpha < T::one(),
      "alpha must be in (0,1)"
    );
    assert!(c > T::zero(), "c must be positive");
    Self {
      backend: Cpu,
      alpha,
      c,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> AlphaStableSubordinator<T, S, B> {}

/// The Euler engine's view of the positive-stable subordinator. Every
/// exponent the transform needs depends on `α` alone, and the scale on `c`
/// and `dt`, so all five travel as parameters and the step is one expression.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for AlphaStableSubordinator<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let one = T::one();
    let dt = self.time_step();
    crate::euler::EulerSpec::StableSubordinator {
      alpha: self.alpha,
      inv_alpha: one / self.alpha,
      one_minus_alpha: one - self.alpha,
      tail_exp: (one - self.alpha) / self.alpha,
      scale: (self.c * dt).powf(one / self.alpha),
      pi: T::from_f64_fast(std::f64::consts::PI),
    }
  }

  fn initial_value(&self) -> T {
    self.x0.unwrap_or(T::zero())
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn device_seed(&self) -> u64 {
    rand::Rng::random(&mut self.seed.rng())
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] AlphaStableSubordinator<T, S> { alpha, c, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for AlphaStableSubordinator<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = AlphaStableSubordinatorSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> AlphaStableSubordinatorSampler<T> {
    let x0 = self.x0.unwrap_or(T::zero());
    let n_increments = self.n.saturating_sub(1).max(1);
    let t_max = self.t.unwrap_or(T::one()).to_f64().unwrap();
    let alpha = self.alpha.to_f64().unwrap();
    let c = self.c.to_f64().unwrap();
    let dt = t_max / n_increments as f64;
    let scale = (c * dt).powf(1.0 / alpha);
    AlphaStableSubordinatorSampler {
      n: self.n,
      x0,
      alpha,
      scale,
      uniform: SimdUniform::<f64>::new(0.0, 1.0, &self.seed),
    }
  }

  /// Through the Euler engine: on a device the transform runs in the kernel,
  /// on the host devices it is this process's own sampler, chunked exactly as
  /// `ProcessExt` chunks.
  fn sample(&self) -> Array1<T> {
    self.backend.euler_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    self.backend.euler_paths_map(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    self.backend.euler_paths(self, m)
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    self.backend.try_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    self.backend.try_euler_paths(self, m)
  }
}

/// Reusable [`AlphaStableSubordinator`] sampling state: the owned uniform
/// source driving the positive-stable increments plus precomputed scales.
#[doc(hidden)]
pub struct AlphaStableSubordinatorSampler<T: FloatExt> {
  n: usize,
  x0: T,
  alpha: f64,
  scale: f64,
  uniform: SimdUniform<f64>,
}

impl<T: FloatExt> AlphaStableSubordinatorSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let mut level = self.x0.to_f64().unwrap();
    for x in out[1..].iter_mut() {
      let s = sample_positive_stable(self.alpha, &self.uniform);
      level += self.scale * s;
      *x = T::from_f64_fast(level);
    }
  }
}

impl<T: FloatExt> PathSampler<T> for AlphaStableSubordinatorSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("AlphaStableSubordinator output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyAlphaStableSubordinator, AlphaStableSubordinator,
  sig: (alpha, c, n, x0=None, t=None, seed=None, dtype=None),
  params: (alpha: f64, c: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
