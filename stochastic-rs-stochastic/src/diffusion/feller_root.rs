//! # FellerRoot
//!
//! $$
//! dX_t=X_t(\theta_1 - X_t(\theta_3^3 - \theta_1\theta_2))\,dt+\theta_3 X_t^{3/2}\,dW_t
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

#[derive(Clone, Copy)]
pub struct FellerRoot<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Coefficient θ₁ setting the linear part of the drift
  /// `X_t(θ_1 − X_t(θ_3³ − θ_1θ_2))`.
  pub theta1: T,
  /// Coefficient θ₂, entering the drift's quadratic term only through the
  /// combination `θ_3³ − θ_1θ_2`.
  pub theta2: T,
  /// Coefficient θ₃: sets the diffusion scale (`θ_3 X_t^{3/2}`) and,
  /// via `θ_3³`, also contributes to the drift's quadratic term.
  pub theta3: T,
  /// Number of points sampled along the Feller-root path.
  pub n: usize,
  /// Initial value X₀ of the Feller-root path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on) or [`on_device`](Self::on_device).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> FellerRoot<T, S> {
  pub fn new(
    theta1: T,
    theta2: T,
    theta3: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self {
      backend: Cpu,
      theta1,
      theta2,
      theta3,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> FellerRoot<T, S, B> {}

/// The Euler engine's view of the Feller root process. The drift's constant
/// `θ₃³ − θ₁θ₂` depends on no state, so it is folded here and travels as one
/// parameter rather than being recomputed on every step of every path.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for FellerRoot<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::FellerRoot {
      theta1: self.theta1,
      decay: self.theta3.powi(3) - self.theta1 * self.theta2,
      theta3: self.theta3,
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

backend_switch!([T: FloatExt, S: SeedExt] FellerRoot<T, S> { theta1, theta2, theta3, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for FellerRoot<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = FellerRootSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> FellerRootSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    FellerRootSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      theta1: self.theta1,
      theta2: self.theta2,
      theta3: self.theta3,
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

/// Reusable [`FellerRoot`] sampling state.
#[doc(hidden)]
pub struct FellerRootSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  theta1: T,
  theta2: T,
  theta3: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> FellerRootSampler<T> {
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
      let drift = prev * (self.theta1 - prev * (self.theta3.powi(3) - self.theta1 * self.theta2));
      let next = prev + drift * self.dt + self.theta3 * prev.abs().powf(T::from_f64_fast(1.5)) * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for FellerRootSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("FellerRoot output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyFellerRoot, FellerRoot,
  sig: (theta1, theta2, theta3, n, x0=None, t=None, seed=None, dtype=None),
  params: (theta1: f64, theta2: f64, theta3: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);
