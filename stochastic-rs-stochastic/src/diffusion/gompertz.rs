//! # Gompertz
//!
//! $$
//! dX_t=aX_t\ln\!\left(\frac{K}{X_t}\right)dt+\sigma X_t dW_t
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

/// Gompertz diffusion
/// dX_t = (a - b ln X_t) X_t dt + sigma X_t dW_t
#[derive(Clone)]
pub struct Gompertz<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Growth-rate coefficient a in the drift `X_t(a - b·ln X_t)dt` — together
  /// with `b`, sets the asymptotic level `exp(a/b)`.
  pub a: T,
  /// Log-growth deceleration coefficient b in the drift — NOT a diffusion
  /// term despite the field's position; `b` multiplies `ln X_t` inside the
  /// drift only. Together with `a`, sets the asymptotic level `exp(a/b)`.
  pub b: T,
  /// Proportional volatility scale σ multiplying `X_t dW_t`.
  pub sigma: T,
  /// Number of points sampled along the Gompertz path.
  pub n: usize,
  /// Initial population/level X₀ (floored at 1e-12 to keep `ln X_t` finite).
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Gompertz<T, S> {
  pub fn new(a: T, b: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    Self {
      backend: Cpu,
      a,
      b,
      sigma,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Gompertz<T, S, B> {}

/// The Euler engine's view of the Gompertz process. `X₀` is floored here as
/// the host sampler floors it, which is what makes the step's own floor the
/// only guard the coefficients need.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Gompertz<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::Gompertz {
      a: self.a,
      b: self.b,
      sigma: self.sigma,
    }
  }

  fn initial_value(&self) -> T {
    self.x0.unwrap_or(T::zero()).max(T::from_f64_fast(1e-12))
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

backend_switch!([T: FloatExt, S: SeedExt] Gompertz<T, S> { a, b, sigma, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for Gompertz<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = GompertzSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> GompertzSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    GompertzSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      a: self.a,
      b: self.b,
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

/// Reusable [`Gompertz`] sampling state.
#[doc(hidden)]
pub struct GompertzSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  a: T,
  b: T,
  diff_scale: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> GompertzSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    let threshold = T::from_f64_fast(1e-12);
    let x0 = self.x0.max(threshold);
    out[0] = x0;
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.normal.fill_slice(tail);
    let mut prev = x0;
    for z in tail.iter_mut() {
      let xi = prev.max(threshold);
      let drift = (self.a - self.b * xi.ln()) * xi * self.dt;
      let diff = self.diff_scale * xi * *z;
      let next = xi + drift + diff;
      let clamped = next.max(threshold);
      *z = clamped;
      prev = clamped;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for GompertzSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("Gompertz output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyGompertz, Gompertz,
  sig: (a, b, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (a: f64, b: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);
