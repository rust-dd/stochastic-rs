//! # Cfgns
//!
//! $$
//! Z_t=L\eta_t^H,\quad \operatorname{Cov}(\eta_i^H,\eta_j^H)=\gamma_H(i-j)
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::fgn::Fgn;
use crate::device::Cpu;
use crate::device::DeviceError;
use crate::device::FgnBackend;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Clone)]
pub struct Cfgns<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Instantaneous correlation ρ between the two output fGn streams.
  pub rho: T,
  /// Number of points sampled along each correlated-fGn stream.
  pub n: usize,
  /// Simulation horizon [0, t] for both streams (defaults to 1 when
  /// omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
  fgn: Fgn<T, Unseeded, B>,
}

impl<T: FloatExt, S: SeedExt> Cfgns<T, S, Cpu> {
  pub fn new(hurst: T, rho: T, n: usize, t: Option<T>, seed: S) -> Self {
    assert!(
      (T::zero()..=T::one()).contains(&hurst),
      "Hurst parameter must be in (0, 1)"
    );
    assert!(
      (-T::one()..=T::one()).contains(&rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    Self {
      hurst,
      rho,
      n,
      t,
      seed,
      fgn: Fgn::new(hurst, n, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> Cfgns<T, S, B> {
  /// Sample with an explicit seed, used by callers like Cfbms.
  pub fn sample_with_seed(&self, seed: u64) -> [Array1<T>; 2] {
    self.sample_impl(&Deterministic::new(seed))
  }

  /// Core sampling — monomorphised per seed strategy, zero runtime branching.
  /// Uses one paired fGN pass (real/imag of a single circulant FFT) for the two
  /// independent fields; on a GPU backend they come from a batch of two.
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

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>> ProcessExt<T>
  for Cfgns<T, S, B>
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = CfgnsSampler<'s, T, S, B>
  where
    Self: 's;

  /// A CPU sampler borrowing the process for its inner [`Fgn`] (`Arc`-shared
  /// FFT plan + eigenvalues) and owning a seed derived once at
  /// construction. Deriving (not cloning) is what decorrelates chunks: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart. The first
  /// `sample` reproduces the legacy `sample_impl(&seed)` stream bit-for-bit
  /// — `sample_impl`'s own use of the seed is what advances it, the same
  /// tick the legacy code consumed from `self.seed` directly — and each
  /// subsequent call advances the owned seed further for an independent
  /// correlated pair.
  fn sampler(&self) -> CfgnsSampler<'_, T, S, B> {
    CfgnsSampler {
      cfgns: self,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine: on a device both rows are produced in the
  /// kernel from the increments the fractional pipeline wrote to that same
  /// device, on the host devices it is this process's own sampler, chunked
  /// exactly as [`ProcessExt`] chunks.
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

/// Reusable [`Cfgns`] sampling state: borrows the process for its inner [`Fgn`]
/// (one paired fGN pass per call) and owns a seed derived once at
/// construction.
#[doc(hidden)]
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

/// The Euler engine's view of correlated fGn: the pair *is* the step, so the
/// two-shock innovation family reports its own noise. Both rows come out of
/// one embedding — they share a Hurst exponent — so the pipeline draws
/// `2 · m` paths in the one batched call and the step reads its second stream
/// from the buffer's next `paths` rows.
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
    Some(crate::euler::FgnSpec {
      sqrt_eigenvalues: self.fgn.sqrt_eigenvalues.as_slice().expect("contiguous"),
      n: self.fgn.n,
      offset: self.fgn.offset,
      hurst: self.fgn.hurst.to_f64().unwrap_or(0.5),
      t: self.fgn.t.unwrap_or(T::one()).to_f64().unwrap_or(1.0),
      streams: 2,
    })
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
