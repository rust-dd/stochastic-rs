//! # Verhulst
//!
//! $$
//! dX_t=rX_t\left(1-\frac{X_t}{K}\right)dt+\sigma X_t dW_t
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

/// Verhulst (logistic) diffusion
/// dX_t = r X_t (1 - X_t / K) dt + sigma X_t dW_t
#[derive(Debug, Clone)]
pub struct Verhulst<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Logistic growth rate r (matches the module header's own r in
  /// `dX_t=rX_t(1−X_t/K)dt+...`).
  pub r: T,
  /// Carrying capacity K, the level `X` saturates toward (not a jump size: the model has none).
  pub k: T,
  /// Proportional diffusion scale σ multiplying `X_t dW_t`.
  pub sigma: T,
  /// Number of points sampled along the Verhulst path.
  pub n: usize,
  /// Initial value; `None` starts at zero.
  pub x0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Clamp the state into `[0, K]` after each step.
  pub clamp: bool,
  /// The seed: [`Unseeded`] or [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Verhulst<T, S> {
  /// Panics unless `n >= 2`, a given `t` is positive and finite, and every parameter is finite.
  pub fn new(
    r: T,
    k: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    clamp: bool,
    seed: S,
  ) -> Self {
    assert!(
      r.is_finite(),
      "r must satisfy `r.is_finite()`, got r = {r:?}"
    );
    assert!(
      k.is_finite(),
      "k must satisfy `k.is_finite()`, got k = {k:?}"
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
      r,
      k,
      sigma,
      n,
      x0,
      t,
      clamp,
      seed,
    }
  }
}

/// The Euler engine's view. Confinement to `[0, K]` belongs to the process, not the step, so it
/// picks the family and is not a parameter.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Verhulst<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    if self.clamp {
      crate::euler::EulerSpec::VerhulstClamped {
        r: self.r,
        k: self.k,
        sigma: self.sigma,
      }
    } else {
      crate::euler::EulerSpec::Verhulst {
        r: self.r,
        k: self.k,
        sigma: self.sigma,
      }
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

backend_switch!([T: FloatExt, S: SeedExt] Verhulst<T, S> { r, k, sigma, n, x0, t, clamp, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Verhulst<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for Verhulst<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = VerhulstSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> VerhulstSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    VerhulstSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      r: self.r,
      k: self.k,
      diff_scale: self.sigma,
      clamp: self.clamp,
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

/// Reusable [`Verhulst`] sampling state: precomputed Euler scales and the owned
/// Gaussian source.
#[doc(hidden)]
#[derive(Debug)]
pub struct VerhulstSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  r: T,
  k: T,
  diff_scale: T,
  clamp: bool,
  normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt> VerhulstSampler<T> {
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
      let xi = prev;
      let drift = self.r * xi * (T::one() - xi / self.k) * self.dt;
      let diff = self.diff_scale * xi * *z;
      let mut next = xi + drift + diff;
      if self.clamp {
        next = next.clamp(T::zero(), self.k);
      }
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for VerhulstSampler<T> {}

impl<T: FloatExt> PathSampler<T> for VerhulstSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("Verhulst output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyVerhulst, Verhulst,
  sig: (r, k, sigma, n, x0=None, t=None, clamp=None, seed=None, dtype=None),
  params: (r: f64, k: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, clamp: Option<bool> => crate::python_params::clamp),
  device
);
