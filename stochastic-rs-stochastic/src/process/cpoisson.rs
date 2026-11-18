//! # Cpoisson
//!
//! $$
//! X_t=\sum_{k=1}^{N_t}Y_k,\quad N_t\sim\mathrm{Poisson}(\lambda t)
//! $$
//!

use std::any::Any;

use ndarray::Array1;
use ndarray::Axis;
use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::poisson::SimdPoisson;

use super::CompoundPaths;
use super::poisson::Poisson;
use crate::device::Cpu;
use crate::params::Horizon;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Debug, Clone)]
pub struct CompoundPoisson<T, D, S: SeedExt = Unseeded, B = Cpu>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Distribution of per-jump sizes `Y_k`.
  pub distribution: D,
  /// Poisson driver defining jump arrival intensity and timeline; under `Events(n)` the rows hold
  /// the origin and the `n` jumps after it.
  pub poisson: Poisson<T>,
  /// Seed: [`Unseeded`] or a [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T, D, S: SeedExt> CompoundPoisson<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(distribution: D, poisson: Poisson<T>, seed: S) -> Self {
    Self {
      backend: Cpu,
      distribution,
      poisson,
      seed,
    }
  }
}

impl<T, D, S: SeedExt, B> CompoundPoisson<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
}

/// The Euler engine's description of `distribution` when it is a law the kernels draw:
/// `SimdNormal`, or `SimdExp` as the exponential size law; any other law stays on the host.
pub(crate) fn device_jump_sizes<T: FloatExt, D: Any>(
  distribution: &D,
) -> Option<crate::euler::JumpSizes<T>> {
  let any: &dyn Any = distribution;
  if let Some(normal) = any.downcast_ref::<SimdNormal<T>>() {
    return Some(crate::euler::JumpSizes::Normal {
      mean: normal.mu(),
      sd: normal.sigma(),
    });
  }
  device_arrival_rate(distribution).map(|rate| crate::euler::JumpSizes::Exponential { rate })
}

/// The rate of `distribution` when it is the `SimdExp` law [`device_jump_sizes`] recognises,
/// which makes an inter-arrival law a Poisson arrival stream the kernels draw.
pub(crate) fn device_arrival_rate<T: FloatExt, D: Any>(distribution: &D) -> Option<T> {
  (distribution as &dyn Any)
    .downcast_ref::<SimdExp<T>>()
    .map(|exp| exp.lambda())
}

/// The jumps of a jump-diffusion: their size law and the seed their arrivals derive from, at the
/// process's own `lambda`; it holds no `Poisson`, so a zero rate draws zero increments.
#[derive(Clone, Debug)]
pub(crate) struct JumpDriver<D, S> {
  pub(crate) law: D,
  pub(crate) seed: S,
}

impl<D, S: SeedExt> JumpDriver<D, S> {
  /// `law`, its arrivals seeded from a child of `seed`.
  pub(crate) fn new(law: D, seed: &S) -> Self {
    Self {
      law,
      seed: seed.clone().derive(),
    }
  }
}

/// `n` grid increments, the first zero: per step, the sum of a `Poisson(lambda·dt)` count of
/// `distribution`'s draws; no draw at all where `lambda·dt` is zero.
pub(crate) fn grid_increments<T, D, S>(
  distribution: &D,
  lambda: T,
  seed: &S,
  n: usize,
  dt: T,
) -> Array1<T>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
  S: SeedExt,
{
  let mut increments = Array1::<T>::zeros(n);
  if n <= 1 {
    return increments;
  }

  let lambda_dt = (lambda * dt).to_f64().unwrap();
  assert!(
    lambda_dt.is_finite(),
    "CompoundPoisson: lambda * dt must be finite (got lambda={}, dt={})",
    lambda.to_f64().unwrap_or(f64::NAN),
    dt.to_f64().unwrap_or(f64::NAN),
  );
  if lambda_dt <= 0.0 {
    return increments;
  }

  let mut poisson = SimdPoisson::<u32>::new(lambda_dt).seeded(seed);
  let mut rng = seed.rng();
  for i in 1..n {
    let jump_count = poisson.sample();
    let mut jump_sum = T::zero();
    for _ in 0..jump_count {
      jump_sum += distribution.sample(&mut rng);
    }
    increments[i] = jump_sum;
  }

  increments
}

impl<T, D, S: SeedExt, B> CompoundPoisson<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Compound-Poisson jump increments on a grid of `n` points: `increments[0] = 0` and
  /// `increments[i]` is the sum of the jump sizes over `((i-1)dt, i*dt]`.
  pub fn sample_grid_increments(&self, n: usize, dt: T) -> Array1<T> {
    grid_increments(&self.distribution, self.poisson.lambda, &self.seed, n, dt)
  }

  /// Multiplicative jump increments on a grid: with per-jump return `Y`, each interval holds
  /// `prod_k(1 + Y_k) - 1`, which keeps several jumps in one `dt` step.
  pub fn sample_grid_relative_increments(&self, n: usize, dt: T) -> Array1<T> {
    grid_relative_increments(&self.distribution, self.poisson.lambda, &self.seed, n, dt)
  }
}

#[inline]
fn relative_jump_from_count<T, D, R>(distribution: &D, jump_count: u32, rng: &mut R) -> T
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
  R: Rng + ?Sized,
{
  let mut factor = T::one();
  for _ in 0..jump_count {
    let y = distribution.sample(rng);
    factor = factor * (T::one() + y);
  }
  factor - T::one()
}

/// [`grid_increments`] for relative jumps `Y`: per step `∏(1 + Y_k) − 1` over its count, the
/// sizes drawn from a stream one seed draw past the counts'.
pub(crate) fn grid_relative_increments<T, D, S>(
  distribution: &D,
  lambda: T,
  seed: &S,
  n: usize,
  dt: T,
) -> Array1<T>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
  S: SeedExt,
{
  let mut increments = Array1::<T>::zeros(n);
  if n <= 1 {
    return increments;
  }

  let lambda_dt = (lambda * dt).to_f64().unwrap();
  assert!(
    lambda_dt.is_finite(),
    "CompoundPoisson: lambda * dt must be finite (got lambda={}, dt={})",
    lambda.to_f64().unwrap_or(f64::NAN),
    dt.to_f64().unwrap_or(f64::NAN),
  );
  if lambda_dt <= 0.0 {
    return increments;
  }

  let mut poisson = SimdPoisson::<u32>::new(lambda_dt).seeded(seed);
  seed.derive(); // skip one to differ from grid_increments
  let mut rng = seed.rng();
  for i in 1..n {
    let jump_count = poisson.sample();
    increments[i] = relative_jump_from_count(distribution, jump_count, &mut rng);
  }

  increments
}

impl<T, D, S: SeedExt, B> CompoundPoisson<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  /// The jump-size law as the Euler engine's kernels draw it, one size per
  /// arrival; `None` for a distribution they do not carry.
  fn device_jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    device_jump_sizes(&self.distribution).and_then(crate::euler::JumpSizes::single)
  }
}

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 3>
  for CompoundPoisson<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::CompoundPoissonEvents {
      lambda: self.poisson.lambda,
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [T::zero(); 4]
  }

  fn grid_points(&self) -> usize {
    let Horizon::Events(n) = self.poisson.horizon else {
      panic!("the Euler engine describes CompoundPoisson's count mode; horizon mode has no grid");
    };
    n + 1
  }

  /// One step is one arrival, so the grid has no horizon of its own; the
  /// waiting times come from the intensity in the step.
  fn horizon(&self) -> T {
    T::one()
  }

  /// One size per step under the kernels' law; no count is declared, since none is read.
  /// [`ProcessExt::sample`] keeps any other law on the host, so asking here bypassed that guard.
  fn jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    Some(self.device_jump_sizes().expect(
      "CompoundPoisson: the jump-size distribution is not a law the Euler engine draws on a \
       device; sample through `ProcessExt`, which keeps it on the host",
    ))
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> CompoundPaths<T> {
    let _ = <Self as crate::euler::EulerSystem<T, 3>>::grid_points(self);
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T, D, S: SeedExt] CompoundPoisson<T, D, S> { distribution, poisson, seed } via euler where  T: FloatExt,  D: Distribution<T> + Send + Sync);

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for CompoundPoisson<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
}

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for CompoundPoisson<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  type Output = CompoundPaths<T>;
  type Sampler<'s>
    = CompoundPoissonSampler<'s, T, D, S>
  where
    Self: 's;

  /// Derives `self.seed` into the sampler, never clones it: the mixed next tick hash-scrambles
  /// chunk `i`'s basis against chunk `i+1`'s, where a raw snapshot leaves them one stride apart.
  fn sampler(&self) -> CompoundPoissonSampler<'_, T, D, S> {
    CompoundPoissonSampler {
      distribution: &self.distribution,
      poisson: &self.poisson,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine when the arrival count is fixed and the jump-size law is one the
  /// kernels draw; anything else stays on the host, chunked as [`ProcessExt`] chunks.
  fn sample(&self) -> CompoundPaths<T> {
    if self.device_ready() {
      crate::euler::EulerBackend::system_sample(&self.backend, self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&CompoundPaths<T>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      crate::euler::EulerBackend::system_paths_map(&self.backend, self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<CompoundPaths<T>> {
    if self.device_ready() {
      crate::euler::EulerBackend::system_paths(&self.backend, self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<CompoundPaths<T>, crate::device::DeviceError> {
    if self.device_ready() {
      crate::euler::EulerBackend::try_system_sample(&self.backend, self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<CompoundPaths<T>>, crate::device::DeviceError> {
    if self.device_ready() {
      crate::euler::EulerBackend::try_system_paths(&self.backend, self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// What a device runs here: a fixed number of arrivals — the
  /// horizon mode has no grid — and sizes under a law the kernels draw.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(matches!(self.poisson.horizon, Horizon::Events(_)) && self.device_jump_sizes().is_some()))
      .then_some("the horizon mode's missing grid, or a size law the kernels do not draw")
  }
}

/// Reusable [`CompoundPoisson`] sampling state: it borrows the jump law and the Poisson driver
/// and owns the seed source. The three buffers are reallocated per call, as the count varies.
#[doc(hidden)]
#[derive(Debug)]
pub struct CompoundPoissonSampler<'a, T, D, S: SeedExt>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  distribution: &'a D,
  poisson: &'a Poisson<T>,
  seed: S,
}

impl<T, D, S: SeedExt> CompoundPoissonSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  fn sample_inner(&mut self) -> CompoundPaths<T> {
    let poisson = self.poisson.sample_impl(&self.seed);
    let mut jumps = Array1::<T>::zeros(poisson.len());
    self.seed.derive(); // skip one so `rng` differs from `poisson`'s stream
    let mut rng = self.seed.rng();
    for i in 1..poisson.len() {
      jumps[i] = self.distribution.sample(&mut rng);
    }

    let mut cum_jupms = jumps.clone();
    cum_jupms.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

    CompoundPaths {
      arrivals: poisson,
      cumulative: cum_jupms,
      jumps,
    }
  }
}

impl<T, D, S: SeedExt> crate::traits::Sealed for CompoundPoissonSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
}

impl<T, D, S: SeedExt> PathSampler<T> for CompoundPoissonSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = CompoundPaths<T>;

  fn sample_into(&mut self, out: &mut CompoundPaths<T>) {
    *out = self.sample_inner();
  }

  fn sample(&mut self) -> CompoundPaths<T> {
    self.sample_inner()
  }
}

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyCompoundPoisson;

#[cfg(test)]
mod tests;
