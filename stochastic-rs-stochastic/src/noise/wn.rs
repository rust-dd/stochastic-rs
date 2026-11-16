//! # WN
//!
//! $$
//! \xi_i\stackrel{iid}{\sim}\mathcal N(0,1)
//! $$
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Debug, Copy, Clone)]
pub struct Wn<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Number of i.i.d. noise samples generated.
  pub n: usize,
  /// Mean `mu` of each draw, zero when `None`.
  pub mu: Option<T>,
  /// Standard deviation of generated noise samples.
  pub sigma: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Wn<T, S> {
  pub fn new(n: usize, mu: Option<T>, sigma: Option<T>, seed: S) -> Self {
    assert!(n >= 1, "n must satisfy `n >= 1`, got n = {n:?}");
    assert!(
      mu.is_none_or(|v| v.is_finite()),
      "mu must satisfy `mu.is_none_or(|v| v.is_finite())`, got mu = {mu:?}"
    );
    assert!(
      sigma.is_none_or(|v| v.is_finite()),
      "sigma must satisfy `sigma.is_none_or(|v| v.is_finite())`, got sigma = {sigma:?}"
    );
    assert!(
      sigma.is_none_or(|s| s >= T::zero()),
      "sigma must satisfy `sigma.is_none_or(|s| s >= T::zero())`, got sigma = {sigma:?}"
    );
    Wn {
      backend: Cpu,
      n,
      mu,
      sigma,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Wn<T, S, B> {
  /// `sigma` as host and device read it; a write past `new` that breaks its rule fails here.
  fn checked_sigma(&self) -> T {
    let sigma = self.sigma.unwrap_or(T::one());
    assert!(
      sigma.is_finite() && sigma >= T::zero(),
      "sigma must satisfy `sigma.is_finite() && sigma >= T::zero()`, got sigma = {sigma:?}"
    );
    sigma
  }
}

/// The Euler engine's view of white noise: every grid point is one draw.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Wn<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::Innovation {
      mean: self.mu.unwrap_or(T::zero()),
      sd: self.checked_sigma(),
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

backend_switch!([T: FloatExt, S: SeedExt] Wn<T, S> { n, mu, sigma, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Wn<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Wn<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = WnSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> WnSampler<T> {
    let mu = self.mu.unwrap_or(T::zero());
    let sigma = self.checked_sigma();
    WnSampler {
      n: self.n,
      mu,
      normal: (sigma != T::zero()).then(|| SimdNormal::<T>::new(mu, sigma).seeded(&self.seed)),
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

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    self.backend.euler_paths_map_view(self, m, f)
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

/// Reusable [`Wn`] sampling state: the owned Gaussian source, `None` at a zero
/// deviation, where every draw is the mean.
#[doc(hidden)]
#[derive(Debug)]
pub struct WnSampler<T: FloatExt> {
  n: usize,
  mu: T,
  normal: Option<Seeded<SimdNormal<T>>>,
}

impl<T: FloatExt> WnSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    let len = self.n.min(out.len());
    if len == 0 {
      return;
    }
    match &mut self.normal {
      Some(normal) => normal.fill_slice(&mut out[..len]),
      None => out[..len].fill(self.mu),
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for WnSampler<T> {}

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
  sig: (n, mu=None, sigma=None, seed=None, dtype=None),
  params: (n: usize, mu: Option<f64>, sigma: Option<f64>),
  device
);
