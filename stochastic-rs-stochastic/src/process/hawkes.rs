//! Self-exciting point process with exponential kernel (Hawkes, 1971), sampled by Ogata thinning.
//!
//! $$
//! \lambda(t) = \mu + \sum_{t_i < t} \alpha \, e^{-\beta(t - t_i)}
//! $$
//!
//! Baseline `mu > 0`, excitation `alpha >= 0`, decay `beta > 0`; stationary if `alpha / beta < 1`.
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::params::Horizon;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Univariate Hawkes process with exponential kernel.
#[derive(Debug, Clone, Copy)]
pub struct Hawkes<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Baseline intensity (immigration rate).
  pub mu: T,
  /// Excitation magnitude (intensity jump per event).
  pub alpha: T,
  /// Exponential decay rate of excitation.
  pub beta: T,
  /// `Events(n)`: the origin and the `n` event times after it; `Until(t)`: the events before `t`.
  pub horizon: Horizon<T>,
  /// Seed: [`Unseeded`] or a [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Hawkes<T, S> {
  /// Panics unless `mu > 0`, `0 ≤ alpha < beta`, all three are finite and an `Until` horizon is
  /// positive and finite.
  pub fn new(mu: T, alpha: T, beta: T, horizon: Horizon<T>, seed: S) -> Self {
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(
      alpha.is_finite(),
      "alpha must satisfy `alpha.is_finite()`, got alpha = {alpha:?}"
    );
    assert!(
      beta.is_finite(),
      "beta must satisfy `beta.is_finite()`, got beta = {beta:?}"
    );
    horizon.assert_valid();
    let hawkes = Hawkes {
      backend: Cpu,
      mu,
      alpha,
      beta,
      horizon,
      seed,
    };
    hawkes.assert_parameters();
    hawkes
  }
}

impl<T: FloatExt, S: SeedExt, B> Hawkes<T, S, B> {
  /// Replace `horizon`, all else unchanged; panics on an `Until` that is not positive and finite.
  pub fn with_horizon(mut self, horizon: Horizon<T>) -> Self {
    horizon.assert_valid();
    self.horizon = horizon;
    self
  }

  /// Checked in `new` and again where a sampler runs, since the fields stay public.
  #[track_caller]
  fn assert_parameters(&self) {
    let (mu, alpha, beta) = (self.mu, self.alpha, self.beta);
    assert!(
      mu > T::zero(),
      "mu must satisfy `mu > T::zero()`, got mu = {mu:?}"
    );
    assert!(
      alpha >= T::zero(),
      "alpha must satisfy `alpha >= T::zero()`, got alpha = {alpha:?}"
    );
    assert!(
      beta > T::zero(),
      "beta must satisfy `beta > T::zero()`, got beta = {beta:?}"
    );
    assert!(
      alpha < beta,
      "alpha must satisfy `alpha < beta`, got alpha = {alpha:?}, beta = {beta:?}"
    );
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Hawkes<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    self.assert_parameters();
    crate::euler::EulerSpec::HawkesEvents {
      mu: self.mu,
      alpha: self.alpha,
      beta: self.beta,
    }
  }

  /// The first event is the origin, with no excess intensity yet.
  fn initial_value(&self) -> T {
    T::zero()
  }

  fn grid_points(&self) -> usize {
    let Horizon::Events(n) = self.horizon else {
      panic!("the Euler engine describes Hawkes's count mode; horizon mode has no grid");
    };
    n + 1
  }

  /// One step is one event, so the grid has no horizon of its own; the waits
  /// come from the intensities in the step.
  fn horizon(&self) -> T {
    T::one()
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let _ = crate::euler::EulerCoefficients::grid_points(self);
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Hawkes<T, S> { mu, alpha, beta, horizon, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Hawkes<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Hawkes<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = HawkesSampler<T>
  where
    Self: 's;

  /// Builds the Ogata-thinning sampler; the parameter check runs here, so a reused sampler pays it
  /// once.
  fn sampler(&self) -> HawkesSampler<T> {
    self.assert_parameters();

    let cap = match self.horizon {
      Horizon::Events(_) => 0,
      Horizon::Until(t_max) => {
        // Expected number of events: μ·T / (1 − α/β)
        let expected = if t_max > T::zero() {
          (self.mu * t_max / (T::one() - self.alpha / self.beta))
            .to_f64()
            .unwrap_or(0.0)
        } else {
          0.0
        };
        if expected.is_finite() && expected > 0.0 {
          (expected.ceil() as usize).saturating_add(1)
        } else {
          1
        }
      }
    };

    HawkesSampler {
      mu: self.mu,
      alpha: self.alpha,
      beta: self.beta,
      rng: self.seed.rng(),
      horizon: self.horizon,
      cap,
    }
  }

  /// Through the Euler engine in count mode, where every step is one event; horizon mode has a
  /// random length and stays on the host, chunked as [`ProcessExt`] chunks.
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

  /// What a device runs here: a fixed number of events — the
  /// horizon mode's length is itself random and has no grid.
  fn device_fallback(&self) -> Option<&'static str> {
    matches!(self.horizon, Horizon::Until(_))
      .then_some("the horizon mode, whose own length is random and has no grid")
  }
}

/// Reusable [`Hawkes`] sampling state: the RNG of Ogata's thinning and the sampling regime. The
/// intensity falls between events, so the bound $\bar\lambda = \mu + S$ after the last is tight.
#[doc(hidden)]
#[derive(Debug)]
pub struct HawkesSampler<T: FloatExt> {
  mu: T,
  alpha: T,
  beta: T,
  rng: SimdRng,
  horizon: Horizon<T>,
  cap: usize,
}

impl<T: FloatExt> HawkesSampler<T> {
  fn sample_count(&mut self, n: usize) -> Array1<T> {
    let mu = self.mu;
    let alpha = self.alpha;
    let beta = self.beta;
    let mut s = T::zero();
    let mut t = T::zero();

    let mut events = Vec::with_capacity(n + 1);
    events.push(T::zero());
    while events.len() <= n {
      let lambda_bar = mu + s;
      // Exp(λ̄) via inverse transform: −ln(U) / λ̄, U ∈ (0,1]
      let u = T::one() - T::sample_uniform_simd(&mut self.rng);
      let dt = -u.ln() / lambda_bar;
      t += dt;

      // Decay self-exciting component to candidate time
      s = s * (-beta * dt).exp();

      // Accept / reject (Ogata thinning)
      let lambda_t = mu + s;
      let d = T::sample_uniform_simd(&mut self.rng);
      if d * lambda_bar <= lambda_t {
        events.push(t);
        s += alpha;
      }
    }

    Array1::from(events)
  }

  fn sample_horizon(&mut self, t_max: T, cap: usize) -> Array1<T> {
    let mu = self.mu;
    let alpha = self.alpha;
    let beta = self.beta;
    let mut s = T::zero();
    let mut t = T::zero();

    let mut events = Vec::with_capacity(cap);
    events.push(T::zero());
    if t_max <= T::zero() {
      return Array1::from(events);
    }

    while t < t_max {
      let lambda_bar = mu + s;
      let u = T::one() - T::sample_uniform_simd(&mut self.rng);
      let dt = -u.ln() / lambda_bar;
      t += dt;

      if t >= t_max {
        break;
      }

      // Decay self-exciting component to candidate time
      s = s * (-beta * dt).exp();

      // Accept / reject
      let lambda_t = mu + s;
      let d = T::sample_uniform_simd(&mut self.rng);
      if d * lambda_bar <= lambda_t {
        events.push(t);
        s += alpha;
      }
    }

    Array1::from(events)
  }
}

impl<T: FloatExt> crate::traits::Sealed for HawkesSampler<T> {}

impl<T: FloatExt> PathSampler<T> for HawkesSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    *out = self.sample();
  }

  fn sample(&mut self) -> Array1<T> {
    match self.horizon {
      Horizon::Events(n) => self.sample_count(n),
      Horizon::Until(t_max) => self.sample_horizon(t_max, self.cap),
    }
  }
}

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyHawkes;

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  #[test]
  fn hawkes_n_mode() {
    let h = Hawkes::new(0.5, 0.3, 1.0, Horizon::Events(50), Deterministic::new(42));
    let events = h.sample();
    assert_eq!(events.len(), 51);
    // Arrival times must be non-decreasing
    for i in 1..events.len() {
      assert!(events[i] >= events[i - 1]);
    }
    // First event is t = 0
    assert_eq!(events[0], 0.0);
  }

  #[test]
  fn hawkes_t_max_mode() {
    let h = Hawkes::new(0.5, 0.3, 1.0, Horizon::Until(100.0), Deterministic::new(42));
    let events = h.sample();
    // Must have at least t₀ = 0
    assert!(!events.is_empty());
    assert_eq!(events[0], 0.0);
    // All events within horizon
    for &t in events.iter() {
      assert!(t <= 100.0);
    }
    // Non-decreasing
    for i in 1..events.len() {
      assert!(events[i] >= events[i - 1]);
    }
  }

  #[test]
  fn hawkes_deterministic() {
    // Two instances built with one seed produce the same first path; on one instance successive
    // `.sample()` calls advance the seed and differ.
    let h1 = Hawkes::new(0.5, 0.3, 1.0, Horizon::Events(20), Deterministic::new(123));
    let h2 = Hawkes::new(0.5, 0.3, 1.0, Horizon::Events(20), Deterministic::new(123));
    assert_eq!(h1.sample(), h2.sample());
  }

  #[test]
  fn hawkes_sample_par() {
    let h = Hawkes::new(0.5, 0.3, 1.0, Horizon::Until(50.0), Unseeded);
    let paths = h.sample_par(10);
    assert_eq!(paths.len(), 10);
    for p in &paths {
      assert!(!p.is_empty());
      assert_eq!(p[0], 0.0);
    }
  }

  #[test]
  #[should_panic(expected = "alpha must satisfy `alpha < beta`, got alpha = 2.0, beta = 2.0")]
  fn a_field_written_after_new_is_checked_where_the_sampler_runs() {
    let mut h = Hawkes::<f64>::new(0.5, 0.3, 2.0, Horizon::Events(10), Unseeded);
    h.alpha = 2.0;
    h.sample();
  }

  #[test]
  fn hawkes_clustering() {
    // With self-excitation, inter-arrival times should cluster
    // (more events bunch together compared to Poisson)
    let h = Hawkes::new(0.5, 0.8, 1.0, Horizon::Until(1000.0), Deterministic::new(7));
    let events = h.sample();
    let n = events.len();
    assert!(n > 1);

    // Compute inter-arrival times
    let mut iat = Vec::with_capacity(n - 1);
    for i in 1..n {
      iat.push(events[i] - events[i - 1]);
    }

    // Coefficient of variation > 1 indicates clustering
    // (Poisson would have CV ≈ 1)
    let mean = iat.iter().sum::<f64>() / iat.len() as f64;
    let var = iat.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / iat.len() as f64;
    let cv = var.sqrt() / mean;
    assert!(cv > 1.0, "Hawkes should exhibit clustering (CV = {cv})");
  }
}
