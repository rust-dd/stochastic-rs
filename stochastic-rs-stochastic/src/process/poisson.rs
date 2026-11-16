//! # Poisson
//!
//! $$
//! \mathbb{P}(N=k)=e^{-\lambda}\frac{\lambda^k}{k!},\ k\in\mathbb N_0
//! $$
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::exp::SimdExp;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::params::Horizon;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Clone, Copy)]
pub struct Poisson<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Jump intensity (expected arrivals per unit time).
  pub lambda: T,
  /// `Events(n)`: the origin and the `n` arrivals after it; `Until(t)`: the arrivals before `t`.
  pub horizon: Horizon<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Poisson::default().with_lambda(4.0)`. No persisted cache:
/// `sampler()` builds its exponential inter-arrival source fresh from
/// `self` every call.
impl<T: FloatExt, S: SeedExt> Poisson<T, S> {
  pub fn new(lambda: T, horizon: Horizon<T>, seed: S) -> Self {
    assert!(
      lambda.is_finite(),
      "lambda must satisfy `lambda.is_finite()`, got lambda = {lambda:?}"
    );
    horizon.assert_valid();
    Poisson {
      backend: Cpu,
      lambda,
      horizon,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Poisson<T, S, B> {
  /// Replace `lambda`, all else unchanged.
  pub fn with_lambda(mut self, lambda: T) -> Self {
    self.lambda = lambda;
    self
  }

  /// Replace `horizon`, all else unchanged; panics on an `Until` that is not positive and finite.
  pub fn with_horizon(mut self, horizon: Horizon<T>) -> Self {
    horizon.assert_valid();
    self.horizon = horizon;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// λ = 2 over 252 points, the crate's trading-year default.
impl<T: FloatExt> Default for Poisson<T, Unseeded> {
  fn default() -> Self {
    Self::new(T::from_f64_fast(2.0), Horizon::Events(251), Unseeded)
  }
}

impl<T: FloatExt, S: SeedExt, B> Poisson<T, S, B> {
  /// Sample with an explicit seed, used by callers like CompoundPoisson.
  pub fn sample_with_seed(&self, seed: u64) -> Array1<T> {
    self.sample_impl(&Deterministic::new(seed))
  }

  /// One-shot sample from any seed source. Retained for callers such as
  /// [`crate::process::cpoisson`] that need the arrival times directly.
  pub(crate) fn sample_impl<S2: SeedExt>(&self, seed: &S2) -> Array1<T> {
    self.sampler_impl(seed).sample()
  }

  /// Build the reusable sampling state from any seed source.
  pub(crate) fn sampler_impl<S2: SeedExt>(&self, seed: &S2) -> PoissonSampler<T> {
    let distr = SimdExp::<T>::new(self.lambda).seeded(seed);
    // A draw no engine reads; skipping it would move every stream seeded after this one.
    seed.next_seed();
    PoissonSampler {
      distr,
      horizon: self.horizon,
      lambda: self.lambda,
    }
  }
}

/// The Euler engine's view of an [`Events`](Horizon::Events) horizon, a running sum of exponential
/// waits; an [`Until`](Horizon::Until) path's length is random, so it never reaches a kernel.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Poisson<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::PoissonArrivals {
      lambda: self.lambda,
    }
  }

  fn initial_value(&self) -> T {
    T::zero()
  }

  fn grid_points(&self) -> usize {
    let Horizon::Events(n) = self.horizon else {
      panic!("the Euler engine describes Poisson's count mode; horizon mode has no grid");
    };
    n + 1
  }

  /// The step is a sum of inter-arrival times and reads no `dt`, so the
  /// horizon is the unit one every grid-free family reports.
  fn horizon(&self) -> T {
    T::one()
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  /// Reads `grid_points()` first so the host devices refuse a horizon-mode
  /// process exactly as a device does, rather than quietly returning the
  /// variable-length path the engine never described.
  fn host_sample(&self) -> Array1<T> {
    let _ = crate::euler::EulerCoefficients::grid_points(self);
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Poisson<T, S> { lambda, horizon, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Poisson<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Poisson<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = PoissonSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> PoissonSampler<T> {
    self.sampler_impl(&self.seed)
  }

  /// Count mode goes through the Euler engine, which on a device sums the
  /// inter-arrival times in the kernel. Horizon mode stays on this
  /// process's own sampler whatever the backend: its output length is the
  /// event count, which no fixed grid expresses.
  fn sample(&self) -> Array1<T> {
    if self.device_ready() {
      self.backend.euler_sample(self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      self.backend.euler_paths_map(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    if self.device_ready() {
      self.backend.euler_paths_map_view(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, |path| f(path.view()))
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    if self.device_ready() {
      self.backend.euler_paths(self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_sample(self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_euler_paths(self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// What a device runs here: an `Events` horizon; an `Until` path's length is random.
  fn device_fallback(&self) -> Option<&'static str> {
    matches!(self.horizon, Horizon::Until(_))
      .then_some("the horizon mode, whose own length is random and has no grid")
  }
}

/// Reusable [`Poisson`] sampling state: the owned exponential inter-arrival
/// source and the precomputed sampling regime.
#[doc(hidden)]
pub struct PoissonSampler<T: FloatExt> {
  distr: Seeded<SimdExp<T>>,
  horizon: Horizon<T>,
  lambda: T,
}

impl<T: FloatExt> PoissonSampler<T> {
  /// Count-mode fill: the origin, then the running sum of one exponential wait per remaining slot.
  fn fill_count(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = T::zero();
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.distr.fill_slice(tail);
    let mut acc = T::zero();
    for x in tail.iter_mut() {
      acc += *x;
      *x = acc;
    }
  }

  /// Horizon-mode: accumulate arrival times in `(0, t_max)`.
  fn sample_horizon(&mut self, t_max: T, lambda: T) -> Array1<T> {
    let expected = if t_max > T::zero() {
      (lambda * t_max).to_f64().unwrap_or(0.0)
    } else {
      0.0
    };
    let cap = if expected.is_finite() && expected > 0.0 {
      (expected.ceil() as usize).saturating_add(1)
    } else {
      1
    };
    let mut poisson = Vec::with_capacity(cap);
    poisson.push(T::zero());
    if t_max <= T::zero() {
      return Array1::from(poisson);
    }

    let mut t = T::zero();
    while t < t_max {
      t += self.distr.sample();
      if t < t_max {
        poisson.push(t);
      }
    }
    Array1::from(poisson)
  }
}

impl<T: FloatExt> crate::traits::Sealed for PoissonSampler<T> {}

impl<T: FloatExt> PathSampler<T> for PoissonSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    match self.horizon {
      Horizon::Events(_) => {
        let slice = out
          .as_slice_mut()
          .expect("Poisson output must be contiguous");
        self.fill_count(slice);
      }
      Horizon::Until(t_max) => {
        *out = self.sample_horizon(t_max, self.lambda);
      }
    }
  }

  fn sample(&mut self) -> Array1<T> {
    match self.horizon {
      Horizon::Events(n) => array1_from_fill(n + 1, |out| self.fill_count(out)),
      Horizon::Until(t_max) => self.sample_horizon(t_max, self.lambda),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn an_event_horizon_and_a_time_horizon_are_two_regimes() {
    let seed = Deterministic::new(2);
    let by_count =
      Poisson::<f64, Deterministic>::new(3.0, Horizon::Events(10), seed.clone()).sample();
    let by_time = Poisson::<f64, Deterministic>::new(3.0, Horizon::Until(2.0), seed).sample();
    assert_eq!(by_count.len(), 11);
    assert!(by_time.iter().all(|t| *t <= 2.0));
  }
}

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyPoisson;
