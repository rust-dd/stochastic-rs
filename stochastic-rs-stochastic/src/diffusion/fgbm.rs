//! Geometric Brownian motion on fractional Brownian motion, by an Euler scheme from `x0`.
//!
//! $$
//! dS_t=\mu S_t\,dt+\sigma S_t\,dB_t^H
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

/// Fractional GBM `dS = μS dt + σS dB^H`. Fields are private, since the fGN driver caches a
/// spectrum of `hurst`, `n` and `t`: read them through getters, change them through `with_*`.
///
/// ```compile_fail,E0616
/// use stochastic_rs_core::simd_rng::Unseeded;
/// use stochastic_rs_stochastic::diffusion::fgbm::Fgbm;
/// let mut p = Fgbm::<f64>::new(0.7, 0.05, 0.2, 10, None, None, Unseeded);
/// p.n = 1000;
/// ```
#[derive(Debug, Clone)]
pub struct Fgbm<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  hurst: T,
  mu: T,
  sigma: T,
  n: usize,
  x0: Option<T>,
  t: Option<T>,
  seed: S,
  fgn: Fgn<T, Unseeded, B>,
}

impl<T: FloatExt, S: SeedExt> Fgbm<T, S, Cpu> {
  fn check_hurst(hurst: T) {
    require!(hurst.is_finite(), hurst);
  }

  /// Panics unless `0 <= hurst <= 1`, `n >= 2`, a given `t` is positive and finite, and every
  /// parameter is finite.
  #[must_use]
  pub fn new(hurst: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    Self::check_hurst(hurst);
    Self::check_mu(mu);
    Self::check_sigma(sigma);
    Self::check_x0(x0);
    check_t(t);
    check_n(n);

    Self {
      hurst,
      mu,
      sigma,
      n,
      x0,
      t,
      seed,
      fgn: Fgn::for_grid(hurst, n, t),
    }
  }

  /// Replace `hurst` and rebuild the cache, so it is defined on `Cpu` (move with `on` afterwards);
  /// panics unless `0 <= hurst <= 1`.
  pub fn with_hurst(mut self, hurst: T) -> Self {
    Self::check_hurst(hurst);
    self.hurst = hurst;
    self.fgn = Fgn::for_grid(hurst, self.n, self.t);
    self
  }

  /// Replace `n`; panics unless `n >= 2`.
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

impl<T: FloatExt, S: SeedExt, B> Fgbm<T, S, B> {
  fn check_mu(mu: T) {
    require!(mu.is_finite(), mu);
  }

  fn check_sigma(sigma: T) {
    require!(sigma.is_finite(), sigma);
  }

  fn check_x0(x0: Option<T>) {
    require!(x0.is_none_or(|v| v.is_finite()), x0);
  }

  /// Replace `mu`; panics unless `mu` is finite.
  pub fn with_mu(mut self, mu: T) -> Self {
    Self::check_mu(mu);
    self.mu = mu;
    self
  }

  /// Replace `sigma`; panics unless `sigma` is finite.
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

  /// Replace the seed strategy's value; `fgn`'s own seed is a never-read dummy and is not touched.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }

  /// Hurst exponent controlling roughness and long-memory.
  pub fn hurst(&self) -> T {
    self.hurst
  }

  /// Constant proportional drift rate μ — fGBM has no mean reversion.
  pub fn mu(&self) -> T {
    self.mu
  }

  /// Diffusion scale σ multiplying `S_t dB_t^H`.
  pub fn sigma(&self) -> T {
    self.sigma
  }

  /// Number of points sampled along the fGBM path.
  pub fn n(&self) -> usize {
    self.n
  }

  /// Initial value S₀ of the fGBM path.
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
  crate::traits::Sealed for Fgbm<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>> ProcessExt<T>
  for Fgbm<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = FgbmSampler<'s, T, S, B>
  where
    Self: 's;

  /// A CPU sampler that borrows the process for its inner [`Fgn`] and owns a seed derived once:
  /// one derive from `self.seed` per chunk, and each call advances the owned seed to a new path.
  fn sampler(&self) -> FgbmSampler<'_, T, S, B> {
    FgbmSampler {
      fgbm: self,
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

/// Reusable [`Fgbm`] sampling state on the process's [`Fgn`] and a seed derived once: an Euler
/// discretisation of `dS = mu S dt + sigma S dB^H` from `x0`.
#[doc(hidden)]
#[derive(Debug)]
pub struct FgbmSampler<'a, T: FloatExt, S: SeedExt, B> {
  fgbm: &'a Fgbm<T, S, B>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> FgbmSampler<'_, T, S, B> {
  fn try_fill_path(&mut self, out: &mut [T]) -> Result<(), DeviceError> {
    if out.is_empty() {
      return Ok(());
    }
    let p = self.fgbm;
    let dt = p.fgn.dt();
    let fgn = p.fgn.try_noise(&self.seed)?;

    out[0] = p.x0.unwrap_or(T::zero());
    let mut prev = out[0];
    for (dst, inc) in out[1..].iter_mut().zip(fgn.iter()) {
      let next = prev + p.mu * prev * dt + p.sigma * prev * *inc;
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

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> crate::traits::Sealed for FgbmSampler<'_, T, S, B> {}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> PathSampler<T> for FgbmSampler<'_, T, S, B> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Fgbm output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.fgbm.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }

  fn try_sample(&mut self) -> Result<Array1<T>, DeviceError> {
    let mut out = Array1::<T>::zeros(self.fgbm.n);
    self.try_fill_path(out.as_slice_mut().expect("Fgbm output must be contiguous"))?;
    Ok(out)
  }
}

/// The Euler engine's view of fractional GBM: the same geometric family a
/// Gaussian process uses, driven by fractional increments.
impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>>
  crate::euler::EulerCoefficients<T> for Fgbm<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::GeometricBrownian {
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

  /// The pipeline that produces this process's increments: the device runs it
  /// and keeps the result in its own buffer.
  fn fgn_spec(&self) -> Option<crate::euler::FgnSpec<'_, T>> {
    Some(self.fgn.fgn_spec(1))
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Fgbm<T, S> { hurst, mu, sigma, n, x0, t, seed } via fgn euler);

py_process_1d!(PyFgbm, Fgbm,
  sig: (hurst, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (hurst: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::Fgbm;
  use crate::device::Cpu;
  use crate::traits::ProcessExt;

  #[test]
  fn fgbm_on_cpu_matches_plain_sample() {
    let mk = || {
      Fgbm::<f64, _>::new(
        0.7,
        0.1,
        0.2,
        256,
        Some(1.0),
        Some(1.0),
        Deterministic::new(7),
      )
    };
    let plain = mk().sample();
    let on_cpu = mk().on::<Cpu>().sample();

    assert_eq!(plain.len(), on_cpu.len());
    for (a, b) in plain.iter().zip(on_cpu.iter()) {
      assert_eq!(a, b);
    }
  }
}
