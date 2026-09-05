//! # Cfbms
//!
//! $$
//! dX_t=L\,dB_t^H,\quad LL^\top=\Sigma
//! $$
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::device::DeviceError;
use crate::device::FgnBackend;
use crate::noise::fgn::Fgn;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct Cfbms<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst parameter (`0 < H < 1`) shared by both components.
  pub hurst: T,
  /// Instantaneous correlation between the two fractional-noise drivers.
  pub rho: T,
  /// Number of discrete time points in each path.
  pub n: usize,
  /// Simulation horizon [0, t] for both paths (defaults to `1` if `None`).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The increment pipeline both rows draw from. One embedding serves the
  /// pair because they share a Hurst exponent — on a device that is one
  /// batched call of `2 · m` paths, not two launches.
  pub(crate) fgn: Fgn<T, Unseeded, B>,
}

impl<T: FloatExt, S: SeedExt> Cfbms<T, S> {
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
      fgn: Fgn::new(hurst, n - 1, t, Unseeded),
    }
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Cfbms<T, S> { hurst, rho, n, t, seed } via fgn euler);

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>> ProcessExt<T>
  for Cfbms<T, S, B>
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = CfbmsSampler<'s, T, S, B>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart.
  fn sampler(&self) -> CfbmsSampler<'_, T, S, B> {
    CfbmsSampler {
      cfbms: self,
      seed: self.seed.derive(),
    }
  }
}

/// Reusable [`Cfbms`] sampling state: borrows the process for its fractional
/// pipeline (which owns non-`Copy` FFT scratch) and owns the seed source so a
/// Monte-Carlo loop reuses both output buffers.
#[doc(hidden)]
pub struct CfbmsSampler<'a, T: FloatExt, S: SeedExt, B> {
  cfbms: &'a Cfbms<T, S, B>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> CfbmsSampler<'_, T, S, B> {
  /// The two correlated fBm rows, or why the device could not produce the
  /// increments. The pair comes out of one fractional pass and the second
  /// row is the Cholesky combination of the two independent streams.
  fn try_fill_paths(&mut self, fbm1: &mut [T], fbm2: &mut [T]) -> Result<(), DeviceError> {
    let n = self.cfbms.n;
    if n == 0 {
      return Ok(());
    }
    let (fgn1, z) = self.cfbms.fgn.try_noise_pair(&self.seed)?;
    let rho = self.cfbms.rho;
    let c = (T::one() - rho.powi(2)).sqrt();
    fbm1[0] = T::zero();
    fbm2[0] = T::zero();
    for i in 1..n {
      fbm1[i] = fbm1[i - 1] + fgn1[i - 1];
      fbm2[i] = fbm2[i - 1] + rho * fgn1[i - 1] + c * z[i - 1];
    }
    Ok(())
  }

  fn fill_paths(&mut self, fbm1: &mut [T], fbm2: &mut [T]) {
    self
      .try_fill_paths(fbm1, fbm2)
      .unwrap_or_else(crate::device::device_panic)
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> PathSampler<T> for CfbmsSampler<'_, T, S, B> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [fbm1, fbm2] = out;
    let (fbm1, fbm2) = (
      fbm1
        .as_slice_mut()
        .expect("Cfbms output must be contiguous"),
      fbm2
        .as_slice_mut()
        .expect("Cfbms output must be contiguous"),
    );
    self.fill_paths(fbm1, fbm2);
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    self
      .try_sample()
      .unwrap_or_else(crate::device::device_panic)
  }

  fn try_sample(&mut self) -> Result<[Array1<T>; 2], DeviceError> {
    let n = self.cfbms.n;
    let mut fbm1 = Array1::<T>::zeros(n);
    let mut fbm2 = Array1::<T>::zeros(n);
    self.try_fill_paths(
      fbm1.as_slice_mut().expect("contiguous"),
      fbm2.as_slice_mut().expect("contiguous"),
    )?;
    Ok([fbm1, fbm2])
  }
}

/// The Euler engine's view of correlated fBm: the two-row fractional family,
/// whose step is the increment itself, so accumulating a correlated pair of
/// fGN streams is one kernel rather than a host pass over the batch.
impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>>
  crate::euler::EulerSystem<T, 2> for Cfbms<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::CorrelatedFractionalMotion { rho: self.rho }
  }

  fn initial_state(&self) -> [T; 4] {
    [T::zero(); 4]
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

  /// Both rows come out of one embedding — they share a Hurst exponent — so
  /// the pipeline draws `2 · m` paths in the one batched call and the step
  /// reads its second stream from the buffer's next `paths` rows.
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

py_process_2x1d!(PyCfbms, Cfbms,
  sig: (hurst, rho, n, t=None, seed=None, dtype=None),
  params: (hurst: f64, rho: f64, n: usize, t: Option<f64>),
  device
);
