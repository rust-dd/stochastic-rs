//! # Cfgns
//!
//! $$
//! Z_t=L\eta_t^H,\quad \operatorname{Cov}(\eta_i^H,\eta_j^H)=\gamma_H(i-j)
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::fgn::Fgn;
use crate::device::Cpu;
use crate::device::DeviceError;
use crate::device::FgnBackend;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Two correlated fractional Gaussian noises `Z = L η^H` on `n` points over `[0, t]`. Fields are
/// private: the fGN driver caches a spectrum of `hurst`, `n` and `t` that the setters rebuild.
///
/// ```compile_fail,E0616
/// use stochastic_rs_core::simd_rng::Unseeded;
/// use stochastic_rs_stochastic::noise::cfgns::Cfgns;
/// let mut p = Cfgns::<f64>::new(0.7, 0.3, 10, None, Unseeded);
/// p.n = 1000;
/// ```
#[derive(Debug, Clone)]
pub struct Cfgns<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  hurst: T,
  rho: T,
  n: usize,
  t: Option<T>,
  seed: S,
  fgn: Fgn<T, Unseeded, B>,
}

impl<T: FloatExt, S: SeedExt> Cfgns<T, S, Cpu> {
  fn check_hurst(hurst: T) {
    require!(hurst.is_finite(), hurst);
    require!((T::zero()..=T::one()).contains(&hurst), hurst);
  }

  fn check_n(n: usize) {
    require!(n >= 1, n);
  }

  /// Panics unless `0 <= hurst <= 1`, `-1 <= rho <= 1`, `n >= 1`, a given `t` is positive and
  /// finite, and every parameter is finite.
  pub fn new(hurst: T, rho: T, n: usize, t: Option<T>, seed: S) -> Self {
    Self::check_hurst(hurst);
    Self::check_rho(rho);
    Self::check_n(n);
    check_t(t);

    Self {
      hurst,
      rho,
      n,
      t,
      seed,
      fgn: Fgn::new(hurst, n, t, Unseeded),
    }
  }

  /// Replace `hurst`; panics unless `0 <= hurst <= 1`.
  /// Rebuilds the cache, so it is defined on `Cpu`; move with `on` afterwards.
  pub fn with_hurst(mut self, hurst: T) -> Self {
    Self::check_hurst(hurst);
    self.hurst = hurst;
    self.fgn = Fgn::new(hurst, self.n, self.t, Unseeded);
    self
  }

  /// Replace `n` and rebuild the cache, so it is defined on `Cpu` (move with `on` afterwards);
  /// panics unless `n >= 1`.
  pub fn with_increments(mut self, n: usize) -> Self {
    Self::check_n(n);
    self.n = n;
    self.fgn = Fgn::new(self.hurst, n, self.t, Unseeded);
    self
  }

  /// Replace `t` and rebuild the cache, so it is defined on `Cpu` (move with `on` afterwards);
  /// panics unless `t`, when given, is positive and finite.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self.fgn = Fgn::new(self.hurst, self.n, t, Unseeded);
    self
  }
}

impl<T: FloatExt, S: SeedExt, B> Cfgns<T, S, B> {
  fn check_rho(rho: T) {
    require!(rho.is_finite(), rho);
    require!((-T::one()..=T::one()).contains(&rho), rho);
  }

  /// Replace `rho`; panics unless `-1 <= rho <= 1`.
  pub fn with_rho(mut self, rho: T) -> Self {
    Self::check_rho(rho);
    self.rho = rho;
    self
  }

  /// Replace the seed strategy's value. `fgn`'s own
  /// seed is a never-read dummy, so this does not touch it.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }

  /// Hurst exponent controlling roughness and long-memory.
  pub fn hurst(&self) -> T {
    self.hurst
  }

  /// Instantaneous correlation ρ between the two output fGn streams.
  pub fn rho(&self) -> T {
    self.rho
  }

  /// Number of points sampled along each correlated-fGn stream.
  pub fn n(&self) -> usize {
    self.n
  }

  /// Simulation horizon [0, t] for both streams (defaults to 1 when
  /// omitted).
  pub fn t(&self) -> Option<T> {
    self.t
  }

  /// [`Unseeded`] or a [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic).
  pub fn seed(&self) -> &S {
    &self.seed
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> Cfgns<T, S, B> {
  /// One paired fGN pass (the real and imaginary parts of a single circulant FFT) gives the two
  /// independent fields; a GPU backend takes them from a batch of two.
  #[inline]
  pub(crate) fn sample_impl<S2: SeedExt>(&self, seed: &S2) -> [Array1<T>; 2] {
    self
      .try_sample_impl(seed)
      .unwrap_or_else(crate::device::device_panic)
  }

  /// [`sample_impl`](Self::sample_impl), the device's error instead of its
  /// panic.
  pub(crate) fn try_sample_impl<S2: SeedExt>(
    &self,
    seed: &S2,
  ) -> Result<[Array1<T>; 2], DeviceError> {
    let (fgn1, z) = self.fgn.try_noise_pair(seed)?;
    let c = (T::one() - self.rho.powi(2)).sqrt();
    let mut fgn2 = Array1::zeros(self.n);
    for i in 0..self.n {
      fgn2[i] = self.rho * fgn1[i] + c * z[i];
    }
    Ok([fgn1, fgn2])
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>>
  crate::traits::Sealed for Cfgns<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>> ProcessExt<T>
  for Cfgns<T, S, B>
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = CfgnsSampler<'s, T, S, B>
  where
    Self: 's;

  /// A CPU sampler that borrows the process for its inner [`Fgn`] and owns a seed derived once,
  /// which hash-scrambles chunk bases; `sample_impl` advances it, so each call gives a new pair.
  fn sampler(&self) -> CfgnsSampler<'_, T, S, B> {
    CfgnsSampler {
      cfgns: self,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine: a device produces both rows in its kernel from the increments the
  /// fractional pipeline wrote there; the host runs this process's own chunked sampler.
  fn sample(&self) -> [Array1<T>; 2] {
    crate::euler::EulerBackend::system_sample(&self.fgn.backend, self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&[Array1<T>; 2]) -> R + Sync) -> Vec<R> {
    crate::euler::EulerBackend::system_paths_map(&self.fgn.backend, self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<[Array1<T>; 2]> {
    crate::euler::EulerBackend::system_paths(&self.fgn.backend, self, m)
  }

  fn try_sample(&self) -> Result<[Array1<T>; 2], DeviceError> {
    crate::euler::EulerBackend::try_system_sample(&self.fgn.backend, self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<[Array1<T>; 2]>, DeviceError> {
    crate::euler::EulerBackend::try_system_paths(&self.fgn.backend, self, m)
  }
}

/// Reusable [`Cfgns`] sampling state: borrows the process for its inner [`Fgn`] (one paired fGN
/// pass per call) and owns a seed derived once.
#[doc(hidden)]
#[derive(Debug)]
pub struct CfgnsSampler<'a, T: FloatExt, S: SeedExt, B> {
  cfgns: &'a Cfgns<T, S, B>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> CfgnsSampler<'_, T, S, B> {
  fn fill_paths(&mut self, fgn1_out: &mut [T], fgn2_out: &mut [T]) {
    let [fgn1, fgn2] = self.cfgns.sample_impl(&self.seed);
    fgn1_out.copy_from_slice(fgn1.as_slice().expect("Cfgns noise 1 must be contiguous"));
    fgn2_out.copy_from_slice(fgn2.as_slice().expect("Cfgns noise 2 must be contiguous"));
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> crate::traits::Sealed
  for CfgnsSampler<'_, T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> PathSampler<T> for CfgnsSampler<'_, T, S, B> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [a, b] = out;
    let fgn1 = a.as_slice_mut().expect("Cfgns output must be contiguous");
    let fgn2 = b.as_slice_mut().expect("Cfgns output must be contiguous");
    self.fill_paths(fgn1, fgn2);
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    self.cfgns.sample_impl(&self.seed)
  }

  fn try_sample(&mut self) -> Result<[Array1<T>; 2], DeviceError> {
    self.cfgns.try_sample_impl(&self.seed)
  }
}

/// The Euler engine's view: the pair is the step, so the two-shock innovation family reports its
/// noise. One embedding gives both rows: `2 · m` paths per call, the second stream `paths` rows on.
impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>>
  crate::euler::EulerSystem<T, 2> for Cfgns<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::CorrelatedInnovation { rho: self.rho }
  }

  fn initial_state(&self) -> [T; 4] {
    [T::zero(); 4]
  }

  /// Every grid point is a draw, so the frame steps before it writes the
  /// first one and each point consumes one increment.
  fn step_first(&self) -> bool {
    true
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn time_step(&self) -> T {
    self.fgn.dt()
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn fgn_spec(&self) -> Option<crate::euler::FgnSpec<'_, T>> {
    Some(self.fgn.fgn_spec(2))
  }

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Cfgns<T, S> { hurst, rho, n, t, seed } via fgn euler);

py_process_2x1d!(PyCfgns, Cfgns,
  sig: (hurst, rho, n, t=None, seed=None, dtype=None),
  params: (hurst: f64, rho: f64, n: usize, t: Option<f64>),
  device
);
