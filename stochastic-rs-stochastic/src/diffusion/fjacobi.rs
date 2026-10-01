//! # Fjacobi
//!
//! $$
//! dX_t=\kappa(\theta-X_t)dt+\sigma\sqrt{X_t(1-X_t)}\,dB_t^H
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::device::DeviceError;
use crate::device::FgnBackend;
use crate::noise::fgn::Fgn;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Fractional Jacobi process `dX = (α − βX) dt + σ√(X(1 − X)) dB^H` on `n`
/// points over `[0, t]`.
///
/// Fields are private: the fGN driver caches a spectrum derived from `hurst`, `n` and `t`, so
/// parameters are read through getters and changed through the cache-rebuilding `with_*` setters.
///
/// ```compile_fail,E0616
/// use stochastic_rs_core::simd_rng::Unseeded;
/// use stochastic_rs_stochastic::diffusion::fjacobi::FJacobi;
/// let mut p = FJacobi::<f64>::new(0.7, 1.0, 2.0, 0.2, 10, None, None, Unseeded);
/// p.n = 1000;
/// ```
#[derive(Clone)]
pub struct FJacobi<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  hurst: T,
  alpha: T,
  beta: T,
  sigma: T,
  n: usize,
  x0: Option<T>,
  t: Option<T>,
  seed: S,
  fgn: Fgn<T, Unseeded, B>,
}

impl<T: FloatExt, S: SeedExt> FJacobi<T, S, Cpu> {
  #[must_use]
  pub fn new(
    hurst: T,
    alpha: T,
    beta: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");
    assert!(alpha > T::zero(), "alpha must be positive");
    assert!(beta > T::zero(), "beta must be positive");
    assert!(sigma > T::zero(), "sigma must be positive");
    assert!(alpha < beta, "alpha must be less than beta");

    Self {
      hurst,
      alpha,
      beta,
      sigma,
      n,
      x0,
      t,
      seed,
      fgn: Self::fgn_for(hurst, n, t),
    }
  }

  /// Shared by `new()` and the `with_*` setters so they cannot drift.
  fn fgn_for(hurst: T, n: usize, t: Option<T>) -> Fgn<T, Unseeded, Cpu> {
    Fgn::new(hurst, n - 1, t, Unseeded)
  }

  /// Replace `hurst`; rebuilds the embedded `fgn`.
  pub fn with_hurst(mut self, hurst: T) -> Self {
    self.hurst = hurst;
    self.fgn = Self::fgn_for(hurst, self.n, self.t);
    self
  }

  /// Replace `alpha`. Panics unless `0 < alpha < beta`, matching `new()`'s
  /// own assertions: to move both drift parameters up, raise `beta` first.
  pub fn with_alpha(mut self, alpha: T) -> Self {
    assert!(alpha > T::zero(), "alpha must be positive");
    assert!(alpha < self.beta, "alpha must be less than beta");
    self.alpha = alpha;
    self
  }

  /// Replace `beta`. Panics unless `0 < alpha < beta`, matching `new()`'s
  /// own assertions: to move both drift parameters down, lower `alpha` first.
  pub fn with_beta(mut self, beta: T) -> Self {
    assert!(beta > T::zero(), "beta must be positive");
    assert!(self.alpha < beta, "alpha must be less than beta");
    self.beta = beta;
    self
  }

  /// Replace `sigma`. Panics if `sigma <= 0`, matching `new()`'s own
  /// assertion.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    assert!(sigma > T::zero(), "sigma must be positive");
    self.sigma = sigma;
    self
  }

  /// Replace the number of simulation steps `n`; rebuilds the embedded
  /// `fgn`. Panics if `n < 2`, matching `new()`'s own assertion.
  pub fn with_steps(mut self, n: usize) -> Self {
    assert!(n >= 2, "n must be at least 2");
    self.n = n;
    self.fgn = Self::fgn_for(self.hurst, n, self.t);
    self
  }

  /// Replace `x0`, all else unchanged.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    self.x0 = x0;
    self
  }

  /// Replace the simulation horizon `t`; rebuilds the embedded `fgn`.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    self.fgn = Self::fgn_for(self.hurst, self.n, t);
    self
  }

  /// Replace the seed strategy's value, all else unchanged. `fgn`'s own
  /// seed is a never-read dummy, so this does not touch it.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

impl<T: FloatExt, S: SeedExt, B> FJacobi<T, S, B> {
  /// Hurst exponent controlling roughness and long-memory.
  pub fn hurst(&self) -> T {
    self.hurst
  }

  /// Drift intercept in `alpha - beta·X`, i.e. κθ in `κ(θ - X)`; it stays below `beta` so the
  /// implied θ = alpha/beta lies in (0, 1), as the Jacobi boundary requires.
  pub fn alpha(&self) -> T {
    self.alpha
  }

  /// Linear-drift slope (mean-reversion speed κ) in `alpha - beta·X`.
  pub fn beta(&self) -> T {
    self.beta
  }

  /// Diffusion scale σ multiplying `√(X_t(1-X_t)) dB_t^H`.
  pub fn sigma(&self) -> T {
    self.sigma
  }

  /// Number of points sampled along the fractional Jacobi path.
  pub fn n(&self) -> usize {
    self.n
  }

  /// Initial value X₀ of the fractional Jacobi path (clamped into [0, 1]).
  pub fn x0(&self) -> Option<T> {
    self.x0
  }

  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub fn t(&self) -> Option<T> {
    self.t
  }

  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub fn seed(&self) -> &S {
    &self.seed
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>> ProcessExt<T>
  for FJacobi<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = FJacobiSampler<'s, T, S, B>
  where
    Self: 's;

  /// A CPU sampler borrowing the process for its inner [`Fgn`] (`Arc`-shared
  /// FFT plan + eigenvalues) and owning a seed derived once at construction.
  /// Deriving (not cloning) is what decorrelates chunks: the derived value
  /// is `self.seed`'s *mixed* next tick, not a raw snapshot, so chunk `i`'s
  /// basis and chunk `i+1`'s basis are hash-scrambled relative to each
  /// other rather than one raw stride apart. `fill_path` then uses this
  /// owned seed *directly* (no further derive) — exactly one derive from
  /// `self.seed` per chunk, matching what the legacy per-call `derive()`
  /// consumed, so the first path reproduces the legacy stream bit-for-bit.
  /// Repeat calls on one sampler advance the same owned seed further, for
  /// an independent path each time.
  fn sampler(&self) -> FJacobiSampler<'_, T, S, B> {
    FJacobiSampler {
      fjacobi: self,
      seed: self.seed.derive(),
    }
  }

  /// The same launch as [`sample_par`](Self::sample_par), mapped. Without
  /// these two the trait's defaults would send a mapped batch through the
  /// host sampler while `sample_par` ran on the device — the same law, but
  /// a different stream and none of the device's speed.
  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    crate::euler::EulerBackend::euler_paths_map(&self.fgn.backend, self, m, f)
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    crate::euler::EulerBackend::euler_paths_map_view(&self.fgn.backend, self, m, f)
  }

  /// `m` paths through the Euler engine: on a device the whole recursion runs
  /// in the kernel from fGN increments, on the host devices it is this
  /// process's own sampler chunked exactly as `ProcessExt` chunks.
  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    crate::euler::EulerBackend::euler_paths(&self.fgn.backend, self, m)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    crate::euler::EulerBackend::try_euler_paths(&self.fgn.backend, self, m)
  }
}

/// Reusable [`FJacobi`] sampling state: borrows the process for its inner
/// [`Fgn`] and owns a seed derived once at construction. The path is an
/// Euler discretisation of `dX = (alpha - beta X) dt + sigma sqrt(X(1 - X))
/// dB^H`, clamped into `[0, 1]`.
#[doc(hidden)]
pub struct FJacobiSampler<'a, T: FloatExt, S: SeedExt, B> {
  fjacobi: &'a FJacobi<T, S, B>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> FJacobiSampler<'_, T, S, B> {
  fn try_fill_path(&mut self, out: &mut [T]) -> Result<(), DeviceError> {
    if out.is_empty() {
      return Ok(());
    }
    let p = self.fjacobi;
    let dt = p.fgn.dt();
    let fgn = p.fgn.try_noise(&self.seed)?;

    out[0] = p.x0.unwrap_or(T::zero());
    let mut prev = out[0];
    for (dst, inc) in out[1..].iter_mut().zip(fgn.iter()) {
      let next = match prev {
        _ if prev <= T::zero() => T::zero(),
        _ if prev >= T::one() => T::one(),
        _ => {
          prev + (p.alpha - p.beta * prev) * dt + p.sigma * (prev * (T::one() - prev)).sqrt() * *inc
        }
      };
      *dst = next;
      prev = next;
    }
    Ok(())
  }

  fn fill_path(&mut self, out: &mut [T]) {
    self
      .try_fill_path(out)
      .unwrap_or_else(crate::device::device_panic)
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> PathSampler<T> for FJacobiSampler<'_, T, S, B> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("FJacobi output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.fjacobi.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }

  fn try_sample(&mut self) -> Result<Array1<T>, DeviceError> {
    let mut out = Array1::<T>::zeros(self.fjacobi.n);
    self.try_fill_path(
      out
        .as_slice_mut()
        .expect("FJacobi output must be contiguous"),
    )?;
    Ok(out)
  }
}

/// The Euler engine's view of fractional Jacobi: the same absorbing recursion,
/// with fractional increments instead of hashed ones.
impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>>
  crate::euler::EulerCoefficients<T> for FJacobi<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::Jacobi {
      alpha: self.alpha,
      beta: self.beta,
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

  /// The pipeline that produces this process's increments: the device runs it
  /// and keeps the result in its own buffer.
  fn fgn_spec(&self) -> Option<crate::euler::FgnSpec<'_, T>> {
    Some(self.fgn.fgn_spec(1))
  }
}

backend_switch!([T: FloatExt, S: SeedExt] FJacobi<T, S> { hurst, alpha, beta, sigma, n, x0, t, seed } via fgn euler);

py_process_1d!(PyFJacobi, FJacobi,
  sig: (hurst, alpha, beta, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (hurst: f64, alpha: f64, beta: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);
