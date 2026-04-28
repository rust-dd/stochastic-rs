//! # Hawkes
//!
//! Self-exciting point process with exponential kernel (Hawkes, 1971).
//!
//! $$
//! \lambda(t) = \mu + \sum_{t_i < t} \alpha \, e^{-\beta(t - t_i)}
//! $$
//!
//! where $\mu > 0$ is the baseline intensity, $\alpha \ge 0$ the excitation
//! magnitude, and $\beta > 0$ the exponential decay rate. Stationarity
//! requires $\alpha / \beta < 1$ (branching ratio strictly less than one).
//!
//! Simulated via Ogata's thinning algorithm with the recursive intensity
//! update that exploits the exponential kernel's Markov property.
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Univariate Hawkes process with exponential kernel.
///
/// # Sampling modes
///
/// | field   | meaning |
/// |---------|---------|
/// | `n`     | fixed event count (including $t_0 = 0$) |
/// | `t_max` | time-horizon — all events in $[0, T]$ |
///
/// Exactly one of `n` or `t_max` must be `Some`.
#[derive(Clone, Copy)]
pub struct Hawkes<T: FloatExt, S: SeedExt = Unseeded> {
  /// Baseline intensity (immigration rate).
  pub mu: T,
  /// Excitation magnitude (intensity jump per event).
  pub alpha: T,
  /// Exponential decay rate of excitation.
  pub beta: T,
  /// Optional fixed number of sampled events (including $t_0 = 0$).
  pub n: Option<usize>,
  /// Optional terminal time for horizon-based sampling.
  pub t_max: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt> Hawkes<T> {
  pub fn new(mu: T, alpha: T, beta: T, n: Option<usize>, t_max: Option<T>) -> Self {
    Hawkes {
      mu,
      alpha,
      beta,
      n,
      t_max,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> Hawkes<T, Deterministic> {
  pub fn seeded(mu: T, alpha: T, beta: T, n: Option<usize>, t_max: Option<T>, seed: u64) -> Self {
    Hawkes {
      mu,
      alpha,
      beta,
      n,
      t_max,
      seed: Deterministic::new(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> Hawkes<T, S> {
  /// Core sampling — Ogata's thinning with recursive exponential-kernel update.
  ///
  /// Between events the intensity is monotone-decreasing, so the upper bound
  /// $\bar\lambda = \mu + S$ (evaluated right after the last accepted event)
  /// is tight and acceptance is efficient.
  pub(crate) fn sample_impl<S2: SeedExt>(&self, seed: &S2) -> Array1<T> {
    assert!(self.mu > T::zero(), "baseline intensity μ must be positive");
    assert!(self.alpha >= T::zero(), "excitation α must be non-negative");
    assert!(self.beta > T::zero(), "decay rate β must be positive");
    assert!(
      self.alpha < self.beta,
      "stationarity requires α < β (branching ratio α/β < 1)"
    );

    let mu = self.mu;
    let alpha = self.alpha;
    let beta = self.beta;
    let mut rng = seed.rng();

    // S tracks the self-exciting component: Σ α·exp(-β·(t - t_i))
    let mut s = T::zero();
    let mut t = T::zero();

    if let Some(n) = self.n {
      let mut events = Vec::with_capacity(n);
      events.push(T::zero());

      if n <= 1 {
        return Array1::from(events);
      }

      while events.len() < n {
        let lambda_bar = mu + s;
        // Exp(λ̄) via inverse transform: −ln(U) / λ̄, U ∈ (0,1]
        let u = T::one() - T::sample_uniform_simd(&mut rng);
        let dt = -u.ln() / lambda_bar;
        t += dt;

        // Decay self-exciting component to candidate time
        s = s * (-beta * dt).exp();

        // Accept / reject (Ogata thinning)
        let lambda_t = mu + s;
        let d = T::sample_uniform_simd(&mut rng);
        if d * lambda_bar <= lambda_t {
          events.push(t);
          s += alpha;
        }
      }

      Array1::from(events)
    } else if let Some(t_max) = self.t_max {
      // Expected number of events: μ·T / (1 − α/β)
      let expected = if t_max > T::zero() {
        (mu * t_max / (T::one() - alpha / beta))
          .to_f64()
          .unwrap_or(0.0)
      } else {
        0.0
      };
      let cap = if expected.is_finite() && expected > 0.0 {
        (expected.ceil() as usize).saturating_add(1)
      } else {
        1
      };
      let mut events = Vec::with_capacity(cap);
      events.push(T::zero());

      if t_max <= T::zero() {
        return Array1::from(events);
      }

      while t < t_max {
        let lambda_bar = mu + s;
        let u = T::one() - T::sample_uniform_simd(&mut rng);
        let dt = -u.ln() / lambda_bar;
        t += dt;

        if t >= t_max {
          break;
        }

        // Decay self-exciting component to candidate time
        s = s * (-beta * dt).exp();

        // Accept / reject
        let lambda_t = mu + s;
        let d = T::sample_uniform_simd(&mut rng);
        if d * lambda_bar <= lambda_t {
          events.push(t);
          s += alpha;
        }
      }

      Array1::from(events)
    } else {
      panic!("n or t_max must be provided");
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Hawkes<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    self.sample_impl(&self.seed)
  }
}

py_process_1d!(PyHawkes, Hawkes,
  sig: (mu, alpha, beta, n=None, t_max=None, seed=None, dtype=None),
  params: (mu: f64, alpha: f64, beta: f64, n: Option<usize>, t_max: Option<f64>)
);

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn hawkes_n_mode() {
    let h = Hawkes::seeded(0.5, 0.3, 1.0, Some(50), None, 42);
    let events = h.sample();
    assert_eq!(events.len(), 50);
    // Arrival times must be non-decreasing
    for i in 1..events.len() {
      assert!(events[i] >= events[i - 1]);
    }
    // First event is t = 0
    assert_eq!(events[0], 0.0);
  }

  #[test]
  fn hawkes_t_max_mode() {
    let h = Hawkes::seeded(0.5, 0.3, 1.0, None, Some(100.0), 42);
    let events = h.sample();
    // Must have at least t₀ = 0
    assert!(events.len() >= 1);
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
    // Inter-instance reproducibility: two separately built instances with the
    // same seed produce the same first path. Intra-instance, successive
    // `.sample()` calls advance the seed and yield different paths.
    let h1 = Hawkes::seeded(0.5, 0.3, 1.0, Some(20), None, 123);
    let h2 = Hawkes::seeded(0.5, 0.3, 1.0, Some(20), None, 123);
    assert_eq!(h1.sample(), h2.sample());
  }

  #[test]
  fn hawkes_sample_par() {
    let h = Hawkes::new(0.5, 0.3, 1.0, None, Some(50.0));
    let paths = h.sample_par(10);
    assert_eq!(paths.len(), 10);
    for p in &paths {
      assert!(p.len() >= 1);
      assert_eq!(p[0], 0.0);
    }
  }

  #[test]
  #[should_panic(expected = "stationarity")]
  fn hawkes_rejects_supercritical() {
    let h = Hawkes::<f64>::new(0.5, 2.0, 1.0, Some(10), None);
    h.sample();
  }

  #[test]
  fn hawkes_clustering() {
    // With self-excitation, inter-arrival times should cluster
    // (more events bunch together compared to Poisson)
    let h = Hawkes::seeded(0.5, 0.8, 1.0, None, Some(1000.0), 7);
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
