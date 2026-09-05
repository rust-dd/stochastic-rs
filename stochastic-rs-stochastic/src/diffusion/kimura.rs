//! # Kimura
//!
//! $$
//! dX_t=aX_t(1-X_t)\,dt+\sigma\sqrt{X_t(1-X_t)}\,dW_t
//! $$
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Kimura / Wright–Fisher diffusion
/// dX_t = a X_t (1 - X_t) dt + sigma sqrt(X_t (1 - X_t)) dW_t
#[derive(Clone)]
pub struct Kimura<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Drift-rate coefficient a (Wright–Fisher selection/growth rate) in
  /// `aX_t(1-X_t)dt`.
  pub a: T,
  /// Diffusion scale σ in `σ√(X_t(1-X_t)) dW_t`.
  pub sigma: T,
  /// Number of points sampled along the Wright–Fisher path.
  pub n: usize,
  /// Initial allele frequency X₀ ∈ [0, 1].
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on) or [`on_device`](Self::on_device).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Kimura<T, S> {
  pub fn new(a: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    Self {
      backend: Cpu,
      a,
      sigma,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Kimura<T, S, B> {}

/// The Euler engine's view of the Kimura diffusion.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Kimura<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::Kimura {
      a: self.a,
      sigma: self.sigma,
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

backend_switch!([T: FloatExt, S: SeedExt] Kimura<T, S> { a, sigma, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for Kimura<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = KimuraSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> KimuraSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    KimuraSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      a: self.a,
      diff_scale: self.sigma,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }

  /// Through the Euler engine: on a device the recursion runs in the kernel,
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

/// Reusable [`Kimura`] sampling state.
#[doc(hidden)]
pub struct KimuraSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  a: T,
  diff_scale: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> KimuraSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.normal.fill_slice(tail);
    let mut prev = self.x0;
    for z in tail.iter_mut() {
      // enforce [0,1] domain when computing coefficients
      let xi = prev.clamp(T::zero(), T::one());
      let sqrt_term = (xi * (T::one() - xi)).sqrt();
      let drift = self.a * xi * (T::one() - xi) * self.dt;
      let diff = self.diff_scale * sqrt_term * *z;
      let mut next = xi + drift + diff;
      next = next.clamp(T::zero(), T::one());
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for KimuraSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("Kimura output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyKimura, Kimura,
  sig: (a, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (a: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);
