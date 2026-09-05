//! # ModifiedCIR
//!
//! $$
//! dX_t=-\kappa X_t\,dt+\sigma\sqrt{1+X_t^2}\,dW_t
//! $$
//!
//! A Cox-Ingersoll-Ross-style mean-reverting process (Cox, Ingersoll,
//! Ross (1985), DOI: 10.2307/1911242) modified to replace the `√X_t`
//! diffusion with the globally Lipschitz, linear-growth `√(1+X_t²)`.
//! Unlike CIR, `X_t` is not constrained to stay non-negative — there is
//! no square-root singularity at the origin to protect against, so the
//! process is well-defined and strong-solution-unique for every real
//! `κ`, `σ` with no Feller-type condition. No single paper naming this
//! exact variant could be identified; it reads as a smooth,
//! always-real-valued cousin of CIR rather than an implementation of a
//! specific published model.
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

#[derive(Clone, Copy)]
pub struct ModifiedCIR<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Mean-reversion speed κ pulling `X_t` toward 0 — this model has no
  /// separate long-run-level field; it always reverts to 0.
  pub kappa: T,
  /// Diffusion scale σ multiplying `√(1+X_t²) dW_t`.
  pub sigma: T,
  /// Number of points sampled along the modified-CIR path.
  pub n: usize,
  /// Initial value X₀ of the modified-CIR path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> ModifiedCIR<T, S> {
  pub fn new(kappa: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    Self {
      backend: Cpu,
      kappa,
      sigma,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> ModifiedCIR<T, S, B> {}

/// The Euler engine's view of the modified CIR process.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for ModifiedCIR<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::ModifiedSquareRoot {
      kappa: self.kappa,
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

backend_switch!([T: FloatExt, S: SeedExt] ModifiedCIR<T, S> { kappa, sigma, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for ModifiedCIR<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = ModifiedCirSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> ModifiedCirSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    ModifiedCirSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      kappa: self.kappa,
      sigma: self.sigma,
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

/// Reusable [`ModifiedCIR`] sampling state: precomputed Euler step and the
/// owned Gaussian source.
#[doc(hidden)]
pub struct ModifiedCirSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  kappa: T,
  sigma: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> ModifiedCirSampler<T> {
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
      let next =
        prev + (-self.kappa * prev) * self.dt + self.sigma * (T::one() + prev * prev).sqrt() * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for ModifiedCirSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("ModifiedCIR output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyModifiedCIR, ModifiedCIR,
  sig: (kappa, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (kappa: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);
