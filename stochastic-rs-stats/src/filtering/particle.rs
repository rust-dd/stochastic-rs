//! Bootstrap / Sequential Importance Resampling (SIR) particle filter.
//!
//! Approximates the filtering distribution $p(x_t | y_{1:t})$ by a weighted
//! particle ensemble $\{x_t^{(i)}, w_t^{(i)}\}_{i=1}^N$. The bootstrap variant
//! uses the prior $p(x_t | x_{t-1})$ as proposal so the importance weight
//! reduces to $w_t^{(i)} \propto p(y_t | x_t^{(i)})$.
//!
//! Reference: Gordon, Salmond, Smith, "Novel Approach to Nonlinear /
//! Non-Gaussian Bayesian State Estimation", IEE Proceedings F, 140(2),
//! 107-113 (1993). DOI: 10.1049/ip-f-2.1993.0015

use std::fmt::Display;

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use rand::RngCore;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::SimdFloatExt;

/// Particle resampling scheme.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResamplingScheme {
  /// Multinomial resampling — independent draws per particle.
  Multinomial,
  /// Systematic resampling — single random offset; lower variance.
  #[default]
  Systematic,
  /// Stratified resampling — one draw per equal-probability stratum.
  Stratified,
}

impl Display for ResamplingScheme {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Multinomial => write!(f, "Multinomial"),
      Self::Systematic => write!(f, "Systematic"),
      Self::Stratified => write!(f, "Stratified"),
    }
  }
}

/// Bootstrap particle filter.
///
/// `transition`: $x_t = f(x_{t-1}, v_t)$ where the user samples the noise
/// internally using the supplied RNG.
///
/// `log_observation_lik`: $\log p(y_t | x_t)$.
pub struct ParticleFilter<F, G>
where
  F: Fn(ArrayView1<f64>, &mut SimdRng) -> Array1<f64>,
  G: Fn(ArrayView1<f64>, ArrayView1<f64>) -> f64,
{
  pub particles: Array2<f64>,
  pub log_weights: Array1<f64>,
  pub transition: F,
  pub log_observation_lik: G,
  pub resampling: ResamplingScheme,
  pub effective_threshold: f64,
  rng: SimdRng,
}

impl<F, G> ParticleFilter<F, G>
where
  F: Fn(ArrayView1<f64>, &mut SimdRng) -> Array1<f64>,
  G: Fn(ArrayView1<f64>, ArrayView1<f64>) -> f64,
{
  /// Initialise with `n_particles` drawn from the user-supplied initial
  /// distribution `init`.
  pub fn new<I>(
    n_particles: usize,
    init: I,
    transition: F,
    log_observation_lik: G,
    seed: u64,
  ) -> Self
  where
    I: Fn(&mut SimdRng) -> Array1<f64>,
  {
    assert!(n_particles >= 1);
    let mut rng = Deterministic::new(seed).rng();
    let first = init(&mut rng);
    let dim = first.len();
    let mut particles = Array2::<f64>::zeros((n_particles, dim));
    for j in 0..dim {
      particles[[0, j]] = first[j];
    }
    for i in 1..n_particles {
      let p = init(&mut rng);
      for j in 0..dim {
        particles[[i, j]] = p[j];
      }
    }
    let log_weights = Array1::<f64>::from_elem(n_particles, -(n_particles as f64).ln());
    Self {
      particles,
      log_weights,
      transition,
      log_observation_lik,
      resampling: ResamplingScheme::default(),
      effective_threshold: 0.5 * n_particles as f64,
      rng,
    }
  }

  /// Mean state across particles, weighted by the current normalised weights.
  pub fn mean_state(&self) -> Array1<f64> {
    let (n, d) = self.particles.dim();
    let max_lw = self
      .log_weights
      .iter()
      .cloned()
      .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = self
      .log_weights
      .iter()
      .map(|&lw| (lw - max_lw).exp())
      .collect();
    let total: f64 = weights.iter().sum();
    let mut m = Array1::<f64>::zeros(d);
    for i in 0..n {
      let w = weights[i] / total;
      for j in 0..d {
        m[j] += w * self.particles[[i, j]];
      }
    }
    m
  }

  /// Effective sample size $1 / \sum w_i^2$ in the natural scale.
  pub fn effective_sample_size(&self) -> f64 {
    let max_lw = self
      .log_weights
      .iter()
      .cloned()
      .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = self
      .log_weights
      .iter()
      .map(|&lw| (lw - max_lw).exp())
      .collect();
    let total: f64 = weights.iter().sum();
    let normalised: Vec<f64> = weights.iter().map(|&w| w / total).collect();
    1.0 / normalised.iter().map(|w| w * w).sum::<f64>()
  }

  /// One filter step on observation `y_t`.
  pub fn step(&mut self, y_t: ArrayView1<f64>) {
    let (n, d) = self.particles.dim();
    let mut new_particles = Array2::<f64>::zeros((n, d));
    for i in 0..n {
      let prev = self.particles.row(i);
      let propagated = (self.transition)(prev, &mut self.rng);
      assert_eq!(propagated.len(), d, "transition must preserve state dim");
      for j in 0..d {
        new_particles[[i, j]] = propagated[j];
      }
    }
    for i in 0..n {
      self.log_weights[i] += (self.log_observation_lik)(new_particles.row(i), y_t);
    }
    self.particles = new_particles;
    self.normalise_log_weights();
    if self.effective_sample_size() < self.effective_threshold {
      self.resample();
    }
  }

  fn normalise_log_weights(&mut self) {
    let max_lw = self
      .log_weights
      .iter()
      .cloned()
      .fold(f64::NEG_INFINITY, f64::max);
    let log_total: f64 = self
      .log_weights
      .iter()
      .map(|&lw| (lw - max_lw).exp())
      .sum::<f64>()
      .ln()
      + max_lw;
    for lw in self.log_weights.iter_mut() {
      *lw -= log_total;
    }
  }

  fn resample(&mut self) {
    let (n, d) = self.particles.dim();
    let max_lw = self
      .log_weights
      .iter()
      .cloned()
      .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = self
      .log_weights
      .iter()
      .map(|&lw| (lw - max_lw).exp())
      .collect();
    let total: f64 = weights.iter().sum();
    let mut cumulative = vec![0.0_f64; n];
    let mut acc = 0.0_f64;
    for i in 0..n {
      acc += weights[i] / total;
      cumulative[i] = acc;
    }
    let n_f = n as f64;
    let mut points = Vec::with_capacity(n);
    match self.resampling {
      ResamplingScheme::Multinomial => {
        for _ in 0..n {
          let u: f64 = f64::sample_uniform_simd(&mut self.rng);
          points.push(u);
        }
      }
      ResamplingScheme::Systematic => {
        let u0: f64 = f64::sample_uniform_simd(&mut self.rng) / n_f;
        for i in 0..n {
          points.push(u0 + i as f64 / n_f);
        }
      }
      ResamplingScheme::Stratified => {
        for i in 0..n {
          let u: f64 = f64::sample_uniform_simd(&mut self.rng) / n_f;
          points.push(i as f64 / n_f + u);
        }
      }
    }
    let mut new_particles = Array2::<f64>::zeros((n, d));
    for (i, &p) in points.iter().enumerate() {
      let mut idx = 0;
      while idx + 1 < n && cumulative[idx] < p {
        idx += 1;
      }
      for j in 0..d {
        new_particles[[i, j]] = self.particles[[idx, j]];
      }
    }
    self.particles = new_particles;
    self.log_weights.fill(-(n as f64).ln());
  }
}

/// Convenience: a Gaussian random-walk transition with diagonal step variance,
/// useful as the `transition` argument when the latent state follows a
/// driftless random walk.
pub fn gaussian_random_walk_transition(
  scales: Array1<f64>,
) -> impl Fn(ArrayView1<f64>, &mut SimdRng) -> Array1<f64> {
  move |prev, rng| {
    let d = prev.len();
    let mut out = Array1::<f64>::zeros(d);
    for j in 0..d {
      let dist = SimdNormal::<f64>::new(0.0, scales[j], &Deterministic::new(rng.next_u64()));
      out[j] = prev[j] + dist.sample_fast();
    }
    out
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  #[test]
  fn particle_filter_tracks_random_walk() {
    let truth_dist = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(1));
    let obs_noise = SimdNormal::<f64>::new(0.0, 0.2, &Deterministic::new(2));
    let n = 200;
    let mut x_true = vec![0.0_f64; n];
    let mut steps = vec![0.0_f64; n];
    let mut obs_buf = vec![0.0_f64; n];
    truth_dist.fill_slice_fast(&mut steps);
    obs_noise.fill_slice_fast(&mut obs_buf);
    for i in 1..n {
      x_true[i] = x_true[i - 1] + steps[i];
    }
    let observations: Vec<f64> = (0..n).map(|i| x_true[i] + obs_buf[i]).collect();
    let scale = 1.0;
    let init = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(3));
    let init_fn = move |rng: &mut SimdRng| {
      let _ = rng;
      let mut a = [0.0_f64];
      init.fill_slice_fast(&mut a);
      Array1::from(vec![a[0]])
    };
    let transition_dist = SimdNormal::<f64>::new(0.0, scale, &Deterministic::new(5));
    let transition = move |prev: ArrayView1<f64>, rng: &mut SimdRng| {
      let _ = rng;
      let mut a = [0.0_f64];
      transition_dist.fill_slice_fast(&mut a);
      Array1::from(vec![prev[0] + a[0]])
    };
    let log_obs = move |x: ArrayView1<f64>, y: ArrayView1<f64>| {
      let z = (y[0] - x[0]) / 0.2;
      -0.5 * z * z - 0.2_f64.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    };
    let mut pf = ParticleFilter::new(500, init_fn, transition, log_obs, 7);
    let mut errs = Vec::new();
    for t in 0..n {
      let y = Array1::from(vec![observations[t]]);
      pf.step(y.view());
      let m = pf.mean_state();
      errs.push((m[0] - x_true[t]).abs());
    }
    let mean_err: f64 = errs.iter().sum::<f64>() / errs.len() as f64;
    assert!(mean_err < 1.0, "mean error {mean_err}");
  }

  #[test]
  fn ess_falls_after_step_with_skewed_likelihood() {
    let init = move |_rng: &mut SimdRng| Array1::from(vec![0.0_f64]);
    let transition_dist = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(11));
    let transition = move |prev: ArrayView1<f64>, _rng: &mut SimdRng| {
      let mut a = [0.0_f64];
      transition_dist.fill_slice_fast(&mut a);
      Array1::from(vec![prev[0] + a[0]])
    };
    let log_obs = |x: ArrayView1<f64>, y: ArrayView1<f64>| {
      let z = (y[0] - x[0]) / 0.05;
      -0.5 * z * z
    };
    let mut pf = ParticleFilter::new(100, init, transition, log_obs, 13);
    pf.effective_threshold = 0.0;
    let y = Array1::from(vec![3.0_f64]);
    let n = pf.particles.nrows() as f64;
    pf.step(y.view());
    assert!(pf.effective_sample_size() < n);
  }
}
