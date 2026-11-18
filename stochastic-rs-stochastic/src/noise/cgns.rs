//! # Cgns
//!
//! $$
//! Z_t=L\varepsilon_t,\quad \varepsilon_t\sim\mathcal N(0,I),\ LL^\top=\Sigma
//! $$
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;

use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Debug, Copy, Clone)]
pub struct Cgns<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Instantaneous correlation ρ between the two output Gaussian streams.
  pub rho: T,
  /// Increments drawn per stream; `dt = t / n`.
  pub n: usize,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// [`Unseeded`] or a [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Cgns<T, S> {
  /// Panics unless `n >= 1`, a given `t` is positive and finite, `-1 <= rho <= 1`, and `rho` is
  /// finite.
  pub fn new(rho: T, n: usize, t: Option<T>, seed: S) -> Self {
    assert!(
      rho.is_finite(),
      "rho must satisfy `rho.is_finite()`, got rho = {rho:?}"
    );
    assert!(n >= 1, "n must satisfy `n >= 1`, got n = {n:?}");
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(
      (-T::one()..=T::one()).contains(&rho),
      "rho must satisfy `(-T::one()..=T::one()).contains(&rho)`, got rho = {rho:?}"
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

impl<T: FloatExt, S: SeedExt, B> Cgns<T, S, B> {
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
    let mut n1 =
      stochastic_rs_distributions::normal::SimdNormal::<T>::new(T::zero(), sqrt_dt).seeded(seed);
    let mut n2 =
      stochastic_rs_distributions::normal::SimdNormal::<T>::new(T::zero(), sqrt_dt).seeded(seed);
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

/// The Euler engine's view: every grid point is one draw, so the launch steps before it writes
/// the first, and the step correlates the second component as the host does.
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
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Cgns<T, S> { rho, n, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Cgns<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Cgns<T, S, B> {
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = CgnsSampler<T, S>
  where
    Self: 's;

  /// Derives `self.seed`, never clones it: a raw clone leaves adjacent chunks' bases one γ stride
  /// apart, where the derived tick hash-scrambles them.
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

  /// Through the Euler engine: a device draws both rows in its kernel, the host runs this
  /// process's own sampler, chunked as `ProcessExt` chunks.
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

/// Reusable [`Cgns`] sampling state: the `Copy` generator and a seed derived once by `sampler()`,
/// so each chunk starts from a hash-mixed basis, as `sample_par` and `sample_map` need.
#[doc(hidden)]
#[derive(Debug)]
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

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for CgnsSampler<T, S> {}

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
  params: (rho: f64, n: usize, t: Option<f64>),
  device
);

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  #[should_panic(
    expected = "rho must satisfy `(-T::one()..=T::one()).contains(&rho)`, got rho = 1.5"
  )]
  fn a_correlation_outside_the_unit_interval_is_named_with_its_value() {
    let _ = Cgns::<f64, Unseeded>::new(1.5, 8, Some(1.0), Unseeded);
  }
}
