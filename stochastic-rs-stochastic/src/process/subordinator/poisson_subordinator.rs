use ndarray::Array1;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::poisson::SimdPoisson;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Poisson subordinator with unit jumps:
/// `N_t` with independent increments `Poisson(lambda * dt)`.
pub struct PoissonSubordinator<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Intensity parameter.
  pub lambda: T,
  /// Number of grid points.
  pub n: usize,
  /// Initial level.
  pub x0: Option<T>,
  /// Horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> PoissonSubordinator<T, S> {
  pub fn new(lambda: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    assert!(lambda > T::zero(), "lambda must be positive");
    Self {
      backend: Cpu,
      lambda,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> PoissonSubordinator<T, S, B> {}

/// The Euler engine's view of the Poisson subordinator: the counting family,
/// whose step is the number of jumps the kernel drew.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for PoissonSubordinator<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::CountingProcess
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

  fn jump_intensity(&self) -> Option<T> {
    (self.lambda > T::zero()).then_some(self.lambda)
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

backend_switch!([T: FloatExt, S: SeedExt] PoissonSubordinator<T, S> { lambda, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for PoissonSubordinator<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = PoissonSubordinatorSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> PoissonSubordinatorSampler<T> {
    let x0 = self.x0.unwrap_or(T::zero());
    let n_increments = self.n.saturating_sub(1).max(1);
    let t_max = self.t.unwrap_or(T::one());
    let dt = t_max / T::from_usize_(n_increments);
    let lambda_dt = (self.lambda * dt).to_f64().unwrap();
    PoissonSubordinatorSampler {
      n: self.n,
      x0,
      poisson: SimdPoisson::<u32>::new(lambda_dt, &self.seed),
      rng: self.seed.rng(),
    }
  }

  /// Through the Euler engine: on a device the increment is drawn in the
  /// kernel, on the host devices it is this process's own sampler, chunked
  /// exactly as `ProcessExt` chunks.
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

/// Reusable [`PoissonSubordinator`] sampling state: the owned Poisson driver
/// and its RNG. Each step adds a `Poisson(lambda * dt)` unit-jump count.
#[doc(hidden)]
pub struct PoissonSubordinatorSampler<T: FloatExt> {
  n: usize,
  x0: T,
  poisson: SimdPoisson<u32>,
  rng: SimdRng,
}

impl<T: FloatExt> PoissonSubordinatorSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    for i in 1..out.len() {
      let k = self.poisson.sample(&mut self.rng) as usize;
      out[i] = out[i - 1] + T::from_usize_(k);
    }
  }
}

impl<T: FloatExt> PathSampler<T> for PoissonSubordinatorSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("PoissonSubordinator output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyPoissonSubordinator, PoissonSubordinator,
  sig: (lambda_, n, x0=None, t=None, seed=None, dtype=None),
  params: (lambda_: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
