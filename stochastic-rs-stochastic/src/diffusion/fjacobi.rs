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
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Fractional Jacobi `dX = (α − βX) dt + σ√(X(1 − X)) dB^H`. Fields are private, since the fGN
/// driver caches a spectrum of `hurst`, `n` and `t`: getters read them, `with_*` changes them.
///
/// ```compile_fail,E0616
/// use stochastic_rs_core::simd_rng::Unseeded;
/// use stochastic_rs_stochastic::diffusion::fjacobi::Fjacobi;
/// let mut p = Fjacobi::<f64>::new(0.7, 1.0, 2.0, 0.2, 10, None, None, Unseeded);
/// p.n = 1000;
/// ```
#[derive(Debug, Clone)]
pub struct Fjacobi<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
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

impl<T: FloatExt, S: SeedExt> Fjacobi<T, S, Cpu> {
  fn check_hurst(hurst: T) {
    require!(hurst.is_finite(), hurst);
  }

  /// Panics unless `alpha > 0`, `beta > 0`, `sigma > 0`, a given `t` is positive and finite,
  /// `n >= 2`, `alpha < beta`, and every parameter is finite.
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
    Self::check_hurst(hurst);
    Self::check_alpha(alpha);
    Self::check_beta(beta);
    Self::check_sigma(sigma);
    Self::check_x0(x0);
    check_t(t);
    check_n(n);
    assert!(
      alpha < beta,
      "alpha must satisfy `alpha < beta`, got alpha = {alpha:?}, beta = {beta:?}"
    );

    Self {
      hurst,
      alpha,
      beta,
      sigma,
      n,
      x0,
      t,
      seed,
      fgn: Fgn::for_grid(hurst, n, t),
    }
  }

  /// Replace `hurst` and rebuild the cache, so it is defined on `Cpu` (move with `on` afterwards);
  /// panics unless `hurst` is finite.
  pub fn with_hurst(mut self, hurst: T) -> Self {
    Self::check_hurst(hurst);
    self.hurst = hurst;
    self.fgn = Fgn::for_grid(hurst, self.n, self.t);
    self
  }

  /// Replace `n`; panics if `n < 2`.
  /// Rebuilds the cache, so it is defined on `Cpu`; move with `on` afterwards.
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self.fgn = Fgn::for_grid(self.hurst, n, self.t);
    self
  }

  /// Replace `t` and rebuild the cache, so it is defined on `Cpu` (move with `on` afterwards);
  /// panics unless `t`, when given, is positive and finite.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self.fgn = Fgn::for_grid(self.hurst, self.n, t);
    self
  }
}

impl<T: FloatExt, S: SeedExt, B> Fjacobi<T, S, B> {
  fn check_alpha(alpha: T) {
    require!(alpha.is_finite(), alpha);
    require!(alpha > T::zero(), alpha);
  }

  fn check_beta(beta: T) {
    require!(beta.is_finite(), beta);
    require!(beta > T::zero(), beta);
  }

  fn check_sigma(sigma: T) {
    require!(sigma.is_finite(), sigma);
    require!(sigma > T::zero(), sigma);
  }

  fn check_x0(x0: Option<T>) {
    require!(x0.is_none_or(|v| v.is_finite()), x0);
  }

  /// Replace `alpha`. Panics unless `0 < alpha < beta`, matching `new()`'s
  /// own assertions: to move both drift parameters up, raise `beta` first.
  pub fn with_alpha(mut self, alpha: T) -> Self {
    Self::check_alpha(alpha);
    assert!(
      alpha < self.beta,
      "alpha must satisfy `alpha < self.beta`, got alpha = {alpha:?}, self.beta = {:?}",
      self.beta
    );
    self.alpha = alpha;
    self
  }

  /// Replace `beta`. Panics unless `0 < alpha < beta`, matching `new()`'s
  /// own assertions: to move both drift parameters down, lower `alpha` first.
  pub fn with_beta(mut self, beta: T) -> Self {
    Self::check_beta(beta);
    assert!(
      self.alpha < beta,
      "beta must satisfy `self.alpha < beta`, got beta = {beta:?}, self.alpha = {:?}",
      self.alpha
    );
    self.beta = beta;
    self
  }

  /// Replace `sigma`. Panics if `sigma <= 0`, matching `new()`'s own
  /// assertion.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    Self::check_sigma(sigma);
    self.sigma = sigma;
    self
  }

  /// Replace `x0`; panics unless `x0` is finite.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    Self::check_x0(x0);
    self.x0 = x0;
    self
  }

  /// Replace the seed strategy's value, all else unchanged. `fgn`'s own
  /// seed is a never-read dummy, so this does not touch it.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }

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

  /// Seed: [`Unseeded`] or a [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic).
  pub fn seed(&self) -> &S {
    &self.seed
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>>
  crate::traits::Sealed for Fjacobi<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>> ProcessExt<T>
  for Fjacobi<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = FjacobiSampler<'s, T, S, B>
  where
    Self: 's;

  /// A CPU sampler that borrows the process for its inner [`Fgn`] and owns a seed derived once:
  /// one derive from `self.seed` per chunk, and each call advances the owned seed to a new path.
  fn sampler(&self) -> FjacobiSampler<'_, T, S, B> {
    FjacobiSampler {
      fjacobi: self,
      seed: self.seed.derive(),
    }
  }

  /// The same launch as [`sample_par`](Self::sample_par), mapped; the trait's defaults would map
  /// a batch on the host sampler, a different stream from the device's `sample_par`.
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

  /// `m` paths through the Euler engine: a device runs the recursion in its kernel from fGN
  /// increments, the host runs this process's own sampler, chunked as `ProcessExt` chunks.
  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    crate::euler::EulerBackend::euler_paths(&self.fgn.backend, self, m)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    crate::euler::EulerBackend::try_euler_paths(&self.fgn.backend, self, m)
  }
}

/// Reusable [`Fjacobi`] sampling state on the process's [`Fgn`] and a seed derived once: an Euler
/// discretisation of `dX = (alpha - beta X) dt + sigma sqrt(X(1 - X)) dB^H`, clamped into `[0, 1]`.
#[doc(hidden)]
#[derive(Debug)]
pub struct FjacobiSampler<'a, T: FloatExt, S: SeedExt, B> {
  fjacobi: &'a Fjacobi<T, S, B>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> FjacobiSampler<'_, T, S, B> {
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

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> crate::traits::Sealed
  for FjacobiSampler<'_, T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> PathSampler<T> for FjacobiSampler<'_, T, S, B> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("Fjacobi output must be contiguous");
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
        .expect("Fjacobi output must be contiguous"),
    )?;
    Ok(out)
  }
}

/// The Euler engine's view of fractional Jacobi: the same absorbing recursion,
/// with fractional increments in place of the kernel's own normals.
impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>>
  crate::euler::EulerCoefficients<T> for Fjacobi<T, S, B>
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

backend_switch!([T: FloatExt, S: SeedExt] Fjacobi<T, S> { hurst, alpha, beta, sigma, n, x0, t, seed } via fgn euler);

py_process_1d!(PyFJacobi, Fjacobi,
  sig: (hurst, alpha, beta, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (hurst: f64, alpha: f64, beta: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);
