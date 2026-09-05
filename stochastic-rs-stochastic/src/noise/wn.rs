//! # WN
//!
//! $$
//! \xi_i\stackrel{iid}{\sim}\mathcal N(0,1)
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
pub struct Wn<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Number of i.i.d. noise samples generated.
  pub n: usize,
  /// Target mean level for generated noise samples.
  pub mean: Option<T>,
  /// Standard deviation of generated noise samples.
  pub std_dev: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Wn<T, S> {
  pub fn new(n: usize, mean: Option<T>, std_dev: Option<T>, seed: S) -> Self {
    Wn {
      backend: Cpu,
      n,
      mean,
      std_dev,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Wn<T, S, B> {}

/// The Euler engine's view of white noise: every grid point is one draw.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Wn<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::Innovation {
      mean: self.mean.unwrap_or(T::zero()),
      sd: self.std_dev.unwrap_or(T::one()),
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
    T::from_usize_(self.n)
  }

  fn time_step(&self) -> T {
    T::one()
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

backend_switch!([T: FloatExt, S: SeedExt] Wn<T, S> { n, mean, std_dev, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Wn<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = WnSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> WnSampler<T> {
    let mean = self.mean.unwrap_or(T::zero());
    let std_dev = self.std_dev.unwrap_or(T::one());
    WnSampler {
      n: self.n,
      normal: SimdNormal::<T>::new(mean, std_dev, &self.seed),
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

/// Reusable [`Wn`] sampling state: the owned Gaussian source. Each path is `n`
/// i.i.d. `N(mean, std_dev^2)` draws.
#[doc(hidden)]
pub struct WnSampler<T: FloatExt> {
  n: usize,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> WnSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    let len = self.n.min(out.len());
    if len == 0 {
      return;
    }
    self.normal.fill_slice(&mut out[..len]);
  }
}

impl<T: FloatExt> PathSampler<T> for WnSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Wn output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyWn, Wn,
  sig: (n, mean=None, std_dev=None, seed=None, dtype=None),
  params: (n: usize, mean: Option<f64>, std_dev: Option<f64>)
);
