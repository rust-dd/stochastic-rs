//! # ThreeHalf
//!
//! $$
//! dX_t=\kappa X_t(\mu-X_t)\,dt+\sigma X_t^{3/2}\,dW_t
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

#[derive(Debug, Clone)]
pub struct ThreeHalf<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Mean-reversion speed parameter.
  pub kappa: T,
  /// Long-run mean level μ the drift `κX_t(μ−X_t)` reverts toward.
  pub mu: T,
  /// Diffusion scale σ multiplying `X_t^{3/2} dW_t`.
  pub sigma: T,
  /// Number of points sampled along the 3/2 path.
  pub n: usize,
  /// Initial value; `None` starts at zero.
  pub x0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// The seed: [`Unseeded`] or [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> ThreeHalf<T, S> {
  /// Panics unless `n >= 2`, a given `t` is positive and finite, and every parameter is finite.
  pub fn new(kappa: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    assert!(
      kappa.is_finite(),
      "kappa must satisfy `kappa.is_finite()`, got kappa = {kappa:?}"
    );
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      x0.is_none_or(|v| v.is_finite()),
      "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = {x0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    Self {
      backend: Cpu,
      kappa,
      mu,
      sigma,
      n,
      x0,
      t,
      seed,
    }
  }
}

/// The Euler engine's view of the 3/2 model.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for ThreeHalf<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::ThreeHalf {
      kappa: self.kappa,
      mu: self.mu,
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
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] ThreeHalf<T, S> { kappa, mu, sigma, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for ThreeHalf<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for ThreeHalf<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = ThreeHalfSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> ThreeHalfSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    ThreeHalfSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      kappa: self.kappa,
      mu: self.mu,
      sigma: self.sigma,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt()).seeded(&self.seed),
    }
  }

  /// Through the Euler engine: a device runs the recursion in its kernel, the host runs this
  /// process's own sampler, chunked as `ProcessExt` chunks.
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

/// Reusable [`ThreeHalf`] sampling state: precomputed Euler step and the owned
/// Gaussian source.
#[doc(hidden)]
#[derive(Debug)]
pub struct ThreeHalfSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  kappa: T,
  mu: T,
  sigma: T,
  normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt> ThreeHalfSampler<T> {
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
      let next = prev
        + self.kappa * prev * (self.mu - prev) * self.dt
        + self.sigma * prev.abs().powf(T::from_f64_fast(1.5)) * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for ThreeHalfSampler<T> {}

impl<T: FloatExt> PathSampler<T> for ThreeHalfSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("ThreeHalf output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyThreeHalf, ThreeHalf,
  sig: (kappa, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (kappa: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);
