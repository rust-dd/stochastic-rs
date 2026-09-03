//! # fOU
//!
//! $$
//! dX_t=\theta(\mu-X_t)\,dt+\sigma\,dB_t^H
//! $$
//!
//! Reference: Cheridito P., Kawaguchi H., Maejima M. (2003) —
//! *Fractional Ornstein-Uhlenbeck Processes*, Electronic Journal of
//! Probability 8, paper 3, 1–14, DOI: 10.1214/EJP.v8-125.
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::device::FgnBackend;
use crate::noise::fgn::Fgn;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct Fou<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Mean-reversion speed.
  pub theta: T,
  /// Long-run mean level.
  pub mu: T,
  /// Diffusion scale σ multiplying `dB_t^H`.
  pub sigma: T,
  /// Number of points sampled along the fOU path.
  pub n: usize,
  /// Initial value X₀ of the fOU path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  fgn: Fgn<T, Unseeded, B>,
}

impl<T: FloatExt, S: SeedExt> Fou<T, S, Cpu> {
  #[must_use]
  pub fn new(
    hurst: T,
    theta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");

    Self {
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      seed,
      fgn: Fgn::new(hurst, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> ProcessExt<T> for Fou<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = FouSampler<'s, T, S, B>
  where
    Self: 's;

  /// A CPU sampler borrowing the process for its inner [`Fgn`] (`Arc`-shared
  /// FFT plan + eigenvalues) and owning a seed derived once at construction.
  /// Deriving (not cloning) is what decorrelates chunks: the derived value
  /// is `self.seed`'s *mixed* next tick, not a raw snapshot, so chunk `i`'s
  /// basis and chunk `i+1`'s basis are hash-scrambled relative to each
  /// other rather than one raw stride apart. `fill_path` then uses this
  /// owned seed *directly* (no further derive) — exactly one derive from
  /// `self.seed` per chunk, matching what the legacy per-call `derive()`
  /// consumed, so the first path reproduces the legacy stream bit-for-bit.
  /// Repeat calls on one sampler advance the same owned seed further, for
  /// an independent path each time.
  fn sampler(&self) -> FouSampler<'_, T, S, B> {
    FouSampler {
      fou: self,
      seed: self.seed.derive(),
    }
  }
}

/// Reusable [`Fou`] sampling state: borrows the process for its inner [`Fgn`]
/// and owns a seed derived once at construction. The path is an Euler
/// discretisation of `dX = theta(mu - X) dt + sigma dB^H` started at `x0`.
#[doc(hidden)]
pub struct FouSampler<'a, T: FloatExt, S: SeedExt, B> {
  fou: &'a Fou<T, S, B>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> FouSampler<'_, T, S, B> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    let p = self.fou;
    let dt = p.fgn.dt();
    let fgn = p.fgn.noise(&self.seed);

    out[0] = p.x0.unwrap_or(T::zero());
    let mut prev = out[0];
    for (dst, inc) in out[1..].iter_mut().zip(fgn.iter()) {
      let next = prev + p.theta * (p.mu - prev) * dt + p.sigma * *inc;
      *dst = next;
      prev = next;
    }
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> PathSampler<T> for FouSampler<'_, T, S, B> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Fou output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.fou.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Fou<T, S> { hurst, theta, mu, sigma, n, x0, t, seed } via fgn);

py_process_1d!(PyFou, Fou,
  sig: (hurst, theta, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (hurst: f64, theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::Fou;
  use crate::traits::ProcessExt;

  #[test]
  #[should_panic(expected = "n must be at least 2")]
  fn fou_requires_at_least_two_points() {
    let _ = Fou::<f64>::new(0.7, 1.0, 0.0, 0.2, 1, Some(0.0), Some(1.0), Unseeded);
  }

  #[test]
  fn fou_sigma_zero_matches_deterministic_euler() {
    let theta = 1.3_f64;
    let mu = 0.8_f64;
    let n = 129_usize;
    let x0 = 0.2_f64;
    let t = 1.0_f64;

    let p = Fou::<f64>::new(0.7, theta, mu, 0.0, n, Some(x0), Some(t), Unseeded);
    let x = p.sample();

    let dt = t / (n as f64 - 1.0);
    let mut expected = x0;
    for i in 1..n {
      expected = expected + theta * (mu - expected) * dt;
      assert!((x[i] - expected).abs() < 1e-12, "mismatch at index {i}");
    }
  }

  #[test]
  fn fou_dt_alignment_holds_for_multiple_grid_sizes() {
    let theta = 0.9_f64;
    let mu = -0.1_f64;
    let x0 = 0.35_f64;
    let hs = [0.55_f64, 0.9_f64];
    let ns = [3_usize, 17, 129, 1000];
    let ts = [0.7_f64, 2.0_f64];

    for &h in &hs {
      for &n in &ns {
        for &t in &ts {
          let p = Fou::<f64>::new(h, theta, mu, 0.0, n, Some(x0), Some(t), Unseeded);
          let x = p.sample();

          let dt = t / (n as f64 - 1.0);
          let mut expected = x0;
          for i in 1..n {
            expected = expected + theta * (mu - expected) * dt;
            assert!(
              (x[i] - expected).abs() < 1e-12,
              "mismatch at i={i}, n={n}, t={t}, h={h}"
            );
          }
        }
      }
    }
  }

  #[test]
  fn fou_sample_is_finite() {
    let p = Fou::<f64>::new(0.65, 1.0, 0.0, 0.5, 256, Some(0.1), Some(1.0), Unseeded);
    let x = p.sample();
    assert_eq!(x.len(), 256);
    assert!(x.iter().all(|v| v.is_finite()));
  }
}
