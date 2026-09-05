//! # GN
//!
//! $$
//! \Delta W_i\sim\mathcal N(0,\Delta t)
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

#[derive(Copy, Clone)]
pub struct Gn<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Number of `N(0, dt)` increments sampled (no leading zero).
  pub n: usize,
  /// Simulation horizon [0, t] that `n` increments span (defaults to 1
  /// when omitted); sets `dt = t / n`.
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Gn::default().with_steps(500)`. No persisted cache: `sampler()`
/// builds its Gaussian source fresh from `self` every call.
impl<T: FloatExt, S: SeedExt> Gn<T, S> {
  pub fn new(n: usize, t: Option<T>, seed: S) -> Self {
    Gn {
      backend: Cpu,
      n,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Gn<T, S, B> {
  /// Replace the number of increments `n`, all else unchanged.
  pub fn with_steps(mut self, n: usize) -> Self {
    self.n = n;
    self
  }

  /// Replace the simulation horizon `t`, all else unchanged.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// t=1, n=252 — one trading year of daily `N(0, dt)` increments (this
/// crate's `Default` convention).
impl<T: FloatExt> Default for Gn<T, Unseeded> {
  fn default() -> Self {
    Self::new(252, Some(T::one()), Unseeded)
  }
}

/// The Euler engine's view of Gaussian noise: the innovation family at zero
/// mean and the grid's own step size.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Gn<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::Innovation {
      mean: T::zero(),
      sd: self.dt().sqrt(),
    }
  }

  fn initial_value(&self) -> T {
    T::zero()
  }

  /// Every grid point is a draw, so the launch steps before writing the
  /// first.
  fn step_first(&self) -> bool {
    true
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn time_step(&self) -> T {
    self.dt()
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Gn<T, S> { n, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Gn<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = GnSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> GnSampler<T> {
    GnSampler {
      n: self.n,
      normal: SimdNormal::<T>::new(T::zero(), self.dt().sqrt(), &self.seed),
    }
  }

  /// Through the Euler engine: on a device the draw happens in the kernel, on
  /// the host devices it is this process's own sampler, chunked exactly as
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

/// Reusable [`Gn`] sampling state: the owned Gaussian source. Each path is `n`
/// i.i.d. `N(0, dt)` increments (no leading zero, unlike [`Bm`]).
#[doc(hidden)]
pub struct GnSampler<T: FloatExt> {
  n: usize,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> GnSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    let len = self.n.min(out.len());
    if len == 0 {
      return;
    }
    self.normal.fill_slice(&mut out[..len]);
  }
}

impl<T: FloatExt> PathSampler<T> for GnSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Gn output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

impl<T: FloatExt, S: SeedExt, B> Gn<T, S, B> {
  pub fn fill_slice(&self, out: &mut [T]) {
    let len = self.n.min(out.len());
    if len == 0 {
      return;
    }
    let std_dev = self.dt().sqrt();
    let normal = SimdNormal::<T>::new(T::zero(), std_dev, &self.seed);
    normal.fill_slice(&mut out[..len]);
  }

  pub fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n)
  }
}

py_process_1d!(PyGn, Gn,
  sig: (n, t=None, seed=None, dtype=None),
  params: (n: usize, t: Option<f64>)
);
