//! # Hyperbolic2
//!
//! $$
//! dX_t=\frac{\sigma^2}{2}\left(\beta-\frac{\gamma X_t}{\sqrt{\delta^2+(X_t-\mu)^2}}\right)dt+\sigma\,dW_t
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
pub struct Hyperbolic2<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Drift-shape coefficient β, the constant term inside the bracket
  /// `β − γX_t/√(δ²+(X_t−μ)²)`.
  pub beta: T,
  /// Drift-shape coefficient γ scaling the restoring term inside the same
  /// bracket.
  pub gamma: T,
  /// Curvature parameter δ > 0 of the hyperbolic restoring term near
  /// `X_t = μ`.
  pub delta: T,
  /// Location parameter μ the restoring term is centered on — not a drift
  /// rate; it enters only through `(X_t − μ)²`.
  pub mu: T,
  /// Diffusion scale σ multiplying `dW_t` (also rescales the drift bracket
  /// by `σ²/2`).
  pub sigma: T,
  /// Number of points sampled along the generalized-hyperbolic path.
  pub n: usize,
  /// Initial value X₀ of the generalized-hyperbolic path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Hyperbolic2<T, S> {
  pub fn new(
    beta: T,
    gamma: T,
    delta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self {
      backend: Cpu,
      beta,
      gamma,
      delta,
      mu,
      sigma,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Hyperbolic2<T, S, B> {}

/// The Euler engine's view of the hyperbolic diffusion. The drift's leading
/// `½σ²` is folded here rather than recomputed on every step.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Hyperbolic2<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::HyperbolicDiffusion {
      beta: self.beta,
      gamma: self.gamma,
      delta: self.delta,
      mu: self.mu,
      sigma: self.sigma,
      half_var: T::from_f64_fast(0.5) * self.sigma * self.sigma,
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

backend_switch!([T: FloatExt, S: SeedExt] Hyperbolic2<T, S> { beta, gamma, delta, mu, sigma, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for Hyperbolic2<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = Hyperbolic2Sampler<T>
  where
    Self: 's;

  fn sampler(&self) -> Hyperbolic2Sampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    Hyperbolic2Sampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      beta: self.beta,
      gamma: self.gamma,
      delta: self.delta,
      mu: self.mu,
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

/// Reusable [`Hyperbolic2`] sampling state.
#[doc(hidden)]
pub struct Hyperbolic2Sampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  beta: T,
  gamma: T,
  delta: T,
  mu: T,
  sigma: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> Hyperbolic2Sampler<T> {
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
      let r = (self.delta * self.delta + (prev - self.mu) * (prev - self.mu)).sqrt();
      let half = T::from_f64_fast(0.5);
      let drift = half * self.sigma * self.sigma * (self.beta - self.gamma * prev / r);
      let next = prev + drift * self.dt + self.sigma * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for Hyperbolic2Sampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("Hyperbolic2 output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyHyperbolic2, Hyperbolic2,
  sig: (beta, gamma, delta, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (beta: f64, gamma: f64, delta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);
