//! # Poisson
//!
//! $$
//! \mathbb{P}(N=k)=e^{-\lambda}\frac{\lambda^k}{k!},\ k\in\mathbb N_0
//! $$
//!

use ndarray::Array1;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::exp::SimdExp;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Clone, Copy)]
pub struct Poisson<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Jump intensity (expected arrivals per unit time).
  pub lambda: T,
  /// Optional fixed number of sampled events.
  pub n: Option<usize>,
  /// Optional terminal time for horizon-based sampling.
  pub t_max: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

#[inline]
fn validate_n_or_tmax<T: FloatExt>(n: Option<usize>, t_max: Option<T>, type_name: &'static str) {
  if n.is_none() && t_max.is_none() {
    panic!("{type_name}: n or t_max must be provided");
  }
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Poisson::default().with_lambda(4.0)`. No persisted cache:
/// `sampler()` builds its exponential inter-arrival source fresh from
/// `self` every call.
impl<T: FloatExt, S: SeedExt> Poisson<T, S> {
  pub fn new(lambda: T, n: Option<usize>, t_max: Option<T>, seed: S) -> Self {
    validate_n_or_tmax(n, t_max, "Poisson");
    Poisson {
      backend: Cpu,
      lambda,
      n,
      t_max,
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

  /// Replace the fixed event count `n`; re-validates that at least one of
  /// `n`/the horizon is still provided, matching `new()`'s own check.
  pub fn with_steps(mut self, n: Option<usize>) -> Self {
    validate_n_or_tmax(n, self.t_max, "Poisson");
    self.n = n;
    self
  }

  /// Replace the terminal-time horizon `t_max`; re-validates that at least
  /// one of the horizon/`n` is still provided, matching `new()`'s own
  /// check.
  pub fn with_horizon(mut self, t_max: Option<T>) -> Self {
    validate_n_or_tmax(self.n, t_max, "Poisson");
    self.t_max = t_max;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// λ=2.0 — a textbook Poisson parameterization. t_max=1, n=252 — one
/// trading year of daily steps (this crate's `Default` convention).
impl<T: FloatExt> Default for Poisson<T, Unseeded> {
  fn default() -> Self {
    Self::new(T::from_f64_fast(2.0), Some(252), Some(T::one()), Unseeded)
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
    let distr = SimdExp::<T>::new(self.lambda, seed);
    let rng = seed.rng();
    let mode = if let Some(n) = self.n {
      PoissonMode::Count { n }
    } else if let Some(t_max) = self.t_max {
      PoissonMode::Horizon {
        t_max,
        lambda: self.lambda,
      }
    } else {
      unreachable!("validate_n_or_tmax ensures at least one of n, t_max is set")
    };
    PoissonSampler { distr, rng, mode }
  }
}

/// The Euler engine's view of the **count mode** only: the running sum of
/// exponential inter-arrival times is a bounded forward recursion, which is
/// what a kernel can step. Horizon mode is not — its output length is the
/// number of events in `(0, t_max)`, not a grid — so [`ProcessExt`] keeps
/// that mode on the host and only reaches the engine when `n` is set.
/// Calling the engine directly with a horizon-mode process is a declaration
/// error and says so.
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
    self
      .n
      .expect("the Euler engine describes Poisson's count mode; horizon mode has no grid")
  }

  /// The step is a sum of inter-arrival times and reads no `dt`, so the
  /// horizon is the unit one every grid-free family reports.
  fn horizon(&self) -> T {
    T::one()
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Poisson<T, S> { lambda, n, t_max, seed } via euler);

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
    if self.n.is_some() {
      self.backend.euler_sample(self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    if self.n.is_some() {
      self.backend.euler_paths(self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    if self.n.is_some() {
      self.backend.try_sample(self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    if self.n.is_some() {
      self.backend.try_euler_paths(self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }
}

/// Sampling regime selected at construction: a fixed event count or a
/// time horizon (variable-length output).
enum PoissonMode<T: FloatExt> {
  Count { n: usize },
  Horizon { t_max: T, lambda: T },
}

/// Reusable [`Poisson`] sampling state: the owned exponential inter-arrival
/// source and the precomputed sampling regime.
#[doc(hidden)]
pub struct PoissonSampler<T: FloatExt> {
  distr: SimdExp<T>,
  rng: SimdRng,
  mode: PoissonMode<T>,
}

impl<T: FloatExt> PoissonSampler<T> {
  /// Count-mode fill: `out[0] = 0`, then the running sum of `n - 1`
  /// exponential inter-arrival times.
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
      t += self.distr.sample(&mut self.rng);
      if t < t_max {
        poisson.push(t);
      }
    }
    Array1::from(poisson)
  }
}

impl<T: FloatExt> PathSampler<T> for PoissonSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    match self.mode {
      PoissonMode::Count { .. } => {
        let slice = out
          .as_slice_mut()
          .expect("Poisson output must be contiguous");
        self.fill_count(slice);
      }
      PoissonMode::Horizon { t_max, lambda } => {
        *out = self.sample_horizon(t_max, lambda);
      }
    }
  }

  fn sample(&mut self) -> Array1<T> {
    match self.mode {
      PoissonMode::Count { n } => array1_from_fill(n, |out| self.fill_count(out)),
      PoissonMode::Horizon { t_max, lambda } => self.sample_horizon(t_max, lambda),
    }
  }
}

py_process_1d!(PyPoisson, Poisson,
  sig: (lambda_, n=None, t_max=None, seed=None, dtype=None),
  params: (lambda_: f64, n: Option<usize>, t_max: Option<f64>)
);
