//! # Cgns
//!
//! $$
//! Z_t=L\varepsilon_t,\quad \varepsilon_t\sim\mathcal N(0,I),\ LL^\top=\Sigma
//! $$
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Copy, Clone)]
pub struct Cgns<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Instantaneous correlation ρ between the two output Gaussian streams.
  pub rho: T,
  /// Number of points sampled along each correlated-Gaussian stream.
  pub n: usize,
  /// Simulation horizon [0, t] for both streams (defaults to 1 when
  /// omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Cgns<T, S> {
  pub fn new(rho: T, n: usize, t: Option<T>, seed: S) -> Self {
    assert!(
      (-T::one()..=T::one()).contains(&rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    Self {
      backend: Cpu,
      rho,
      n,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Cgns<T, S, B> {}

impl<T: FloatExt, S: SeedExt, B> Cgns<T, S, B> {
  /// Sample with an explicit seed, used by callers like Cbms.
  pub fn sample_with_seed(&self, seed: u64) -> [Array1<T>; 2] {
    self.sample_impl(&Deterministic::new(seed))
  }

  /// Core sampling — monomorphised per seed strategy, zero runtime branching.
  #[inline]
  pub(crate) fn sample_impl<S2: SeedExt>(&self, seed: &S2) -> [Array1<T>; 2] {
    let mut gn1 = Array1::<T>::zeros(self.n);
    let mut z = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return [gn1, z];
    }

    let sqrt_dt = (self.t.unwrap_or(T::one()) / T::from_usize_(self.n)).sqrt();
    let gn1_slice = gn1.as_slice_mut().expect("Cgns noise 1 must be contiguous");
    let z_slice = z.as_slice_mut().expect("Cgns noise 2 must be contiguous");
    let n1 = stochastic_rs_distributions::normal::SimdNormal::<T>::new(T::zero(), sqrt_dt, seed);
    let n2 = stochastic_rs_distributions::normal::SimdNormal::<T>::new(T::zero(), sqrt_dt, seed);
    n1.fill_slice(gn1_slice);
    n2.fill_slice(z_slice);
    let c = (T::one() - self.rho.powi(2)).sqrt();
    let mut gn2 = Array1::zeros(self.n);

    for i in 0..self.n {
      gn2[i] = self.rho * gn1[i] + c * z[i];
    }

    [gn1, gn2]
  }

  pub fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n)
  }
}

/// The Euler engine's view of a correlated Gaussian pair: every grid point is
/// one draw, so the launch steps before writing the first, and the second
/// component is correlated in the step exactly as the host correlates it.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for Cgns<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::CorrelatedInnovation { rho: self.rho }
  }

  fn initial_state(&self) -> [T; 4] {
    [T::zero(); 4]
  }

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
    self.dt()
  }

  fn device_seed(&self) -> u64 {
    rand::Rng::random(&mut self.seed.rng())
  }

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Cgns<T, S> { rho, n, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Cgns<T, S, B> {
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = CgnsSampler<T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed`: a whole-struct `self.clone()` here
  /// would copy `self.seed`'s raw, unmixed counter into every chunk's
  /// sampler, so adjacent chunks' bases would differ by exactly one γ
  /// stride instead of being hash-scrambled relative to each other — the
  /// same cross-chunk correlation bug this fixes elsewhere in the crate.
  fn sampler(&self) -> CgnsSampler<T, S> {
    CgnsSampler {
      cgns: Cgns {
        backend: Cpu,
        rho: self.rho,
        n: self.n,
        t: self.t,
        seed: self.seed.derive(),
      },
    }
  }

  /// Through the Euler engine: on a device both rows are drawn in the kernel,
  /// on the host devices it is this process's own sampler, chunked exactly as
  /// `ProcessExt` chunks.
  fn sample(&self) -> [Array1<T>; 2] {
    self.backend.system_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&[Array1<T>; 2]) -> R + Sync) -> Vec<R> {
    self.backend.system_paths_map(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<[Array1<T>; 2]> {
    self.backend.system_paths(self, m)
  }

  fn try_sample(&self) -> Result<[Array1<T>; 2], crate::device::DeviceError> {
    self.backend.try_system_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<[Array1<T>; 2]>, crate::device::DeviceError> {
    self.backend.try_system_paths(self, m)
  }
}

/// Reusable [`Cgns`] sampling state: owns the (cheap, `Copy`) generator and a
/// **derived** copy of its seed source. `sampler()` calls `self.seed.derive()`
/// once, so each chunk's `CgnsSampler` starts from a hash-mixed, chunk-unique
/// basis rather than a raw clone — required for `sample_par`/`sample_map` to
/// produce independent chunks. This means the very first `sample()` no
/// longer matches a bare `sample_impl(&seed)` call bit-for-bit (one extra
/// hash-mixing hop versus the pre-existing behavior); no golden test pins
/// that old equivalence. Every subsequent call was already independent
/// under the previous scheme and remains so here.
#[doc(hidden)]
pub struct CgnsSampler<T: FloatExt, S: SeedExt> {
  cgns: Cgns<T, S>,
}

impl<T: FloatExt, S: SeedExt> CgnsSampler<T, S> {
  fn fill_paths(&mut self, gn1_out: &mut [T], gn2_out: &mut [T]) {
    let [gn1, gn2] = self.cgns.sample_impl(&self.cgns.seed);
    gn1_out.copy_from_slice(gn1.as_slice().expect("Cgns noise 1 must be contiguous"));
    gn2_out.copy_from_slice(gn2.as_slice().expect("Cgns noise 2 must be contiguous"));
  }
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for CgnsSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [a, b] = out;
    let gn1 = a.as_slice_mut().expect("Cgns output must be contiguous");
    let gn2 = b.as_slice_mut().expect("Cgns output must be contiguous");
    self.fill_paths(gn1, gn2);
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    self.cgns.sample_impl(&self.cgns.seed)
  }
}

py_process_2x1d!(PyCgns, Cgns,
  sig: (rho, n, t=None, seed=None, dtype=None),
  params: (rho: f64, n: usize, t: Option<f64>)
);
