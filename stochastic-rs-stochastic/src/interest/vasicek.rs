//! # Vasicek
//!
//! $$
//! dr_t=a(b-r_t)dt+\sigma dW_t
//! $$
//!
//! References:
//! - Vasicek O. (1977) — *An Equilibrium Characterization of the Term
//!   Structure*, Journal of Financial Economics 5(2), 177–188,
//!   DOI: 10.1016/0304-405X(77)90016-2.
//! - Uhlenbeck G. E., Ornstein L. S. (1930) — *On the Theory of the
//!   Brownian Motion*, Physical Review 36(5), 823–841,
//!   DOI: 10.1103/PhysRev.36.823 — the underlying mean-reverting
//!   diffusion ([`Ou`]) this file wraps under
//!   short-rate parameter names.
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::device::HostBackend;
use crate::diffusion::ou::Ou;
use crate::diffusion::ou::OuSampler;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Vasicek short-rate model — internally wraps [`Ou`] with the same parameter
/// semantics: in the SDE `dr = a(b − r) dt + σ dW` (file header) the Rust
/// field [`theta`](Self::theta) corresponds to `a` (mean-reversion speed)
/// and [`mu`](Self::mu) corresponds to `b` (long-run mean level).
#[derive(Clone)]
pub struct Vasicek<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Mean-reversion speed (`a` in the SDE). Controls how fast `r` is pulled
  /// back toward [`mu`](Self::mu).
  pub theta: T,
  /// Long-run mean level (`b` in the SDE). The value `r` reverts to as
  /// `t → ∞`.
  pub mu: T,
  /// Diffusion scale σ multiplying `dW_t` (`σ` in the SDE).
  pub sigma: T,
  /// Number of points sampled along the Vasicek path.
  pub n: usize,
  /// Initial short rate r₀.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  ou: Ou<T, S>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on) or [`on_device`](Self::on_device).
  pub backend: B,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Vasicek::default().with_theta(1.0).with_sigma(0.05)`.
///
/// **Cache note, different from every `Cgns`-based type in this crate**:
/// the embedded `ou: Ou<T, S>` is built once in `new()` via
/// `Ou::new(.., seed.derive())` — a one-time derive off the constructor's
/// own `seed` argument. Rebuilding `ou` inside a setter by deriving from
/// `self.seed` *again* would advance `self.seed`'s state a second time,
/// producing a *different* child than a fresh `Vasicek::new(new_field,
/// .., seed)` would (which derives its *first* child from an unadvanced
/// `seed`). So every setter below except `with_seed` rebuilds `ou` by
/// reusing `self.ou.seed.clone()` (the already-fixed derived seed,
/// `Ou::seed` being `pub`) instead of deriving again; only `with_seed`
/// re-derives, from the *new* outer seed, exactly mirroring `new()`'s own
/// construction order.
impl<T: FloatExt, S: SeedExt> Vasicek<T, S> {
  pub fn new(theta: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    Self {
      backend: Cpu,
      mu,
      sigma,
      theta,
      n,
      x0,
      t,
      ou: Ou::new(theta, mu, sigma, n, x0, t, seed.derive()),
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Vasicek<T, S, B> {
  /// Replace `theta`; rebuilds the embedded `Ou`.
  pub fn with_theta(mut self, theta: T) -> Self {
    self.theta = theta;
    let ou_seed = self.ou.seed.clone();
    self.ou = Ou::new(theta, self.mu, self.sigma, self.n, self.x0, self.t, ou_seed);
    self
  }

  /// Replace `mu`; rebuilds the embedded `Ou`.
  pub fn with_mu(mut self, mu: T) -> Self {
    self.mu = mu;
    let ou_seed = self.ou.seed.clone();
    self.ou = Ou::new(self.theta, mu, self.sigma, self.n, self.x0, self.t, ou_seed);
    self
  }

  /// Replace `sigma`; rebuilds the embedded `Ou`.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    self.sigma = sigma;
    let ou_seed = self.ou.seed.clone();
    self.ou = Ou::new(self.theta, self.mu, sigma, self.n, self.x0, self.t, ou_seed);
    self
  }

  /// Replace `x0`; rebuilds the embedded `Ou`.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    self.x0 = x0;
    let ou_seed = self.ou.seed.clone();
    self.ou = Ou::new(self.theta, self.mu, self.sigma, self.n, x0, self.t, ou_seed);
    self
  }

  /// Replace the number of simulation steps `n`; rebuilds the embedded
  /// `Ou`.
  pub fn with_steps(mut self, n: usize) -> Self {
    self.n = n;
    let ou_seed = self.ou.seed.clone();
    self.ou = Ou::new(self.theta, self.mu, self.sigma, n, self.x0, self.t, ou_seed);
    self
  }

  /// Replace the simulation horizon `t`; rebuilds the embedded `Ou`.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    let ou_seed = self.ou.seed.clone();
    self.ou = Ou::new(self.theta, self.mu, self.sigma, self.n, self.x0, t, ou_seed);
    self
  }

  /// Replace the seed strategy's value; re-derives the embedded `Ou`'s
  /// seed from the *new* outer seed, exactly mirroring `new()`'s own
  /// construction order (derive before moving `seed` into `self.seed`).
  pub fn with_seed(mut self, seed: S) -> Self {
    self.ou = Ou::new(
      self.theta,
      self.mu,
      self.sigma,
      self.n,
      self.x0,
      self.t,
      seed.derive(),
    );
    self.seed = seed;
    self
  }
}

/// a=3.0, b=0.03, σ=0.02, r₀=0.03 — a textbook Vasicek parameterization.
/// t=1, n=252 — one trading year of daily steps (this crate's `Default`
/// convention).
impl<T: FloatExt> Default for Vasicek<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(3.0),
      T::from_f64_fast(0.03),
      T::from_f64_fast(0.02),
      252,
      Some(T::from_f64_fast(0.03)),
      Some(T::one()),
      Unseeded,
    )
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Vasicek<T, S> { theta, mu, sigma, n, x0, t, seed, ou } via host);

impl<T: FloatExt, S: SeedExt, B: HostBackend> ProcessExt<T> for Vasicek<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = VasicekSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> VasicekSampler<T> {
    VasicekSampler {
      ou: self.ou.sampler(),
    }
  }
}

/// Reusable [`Vasicek`] sampling state — owns the inner [`OuSampler`], so the
/// Vasicek path is the wrapped OU path with identical parameter semantics.
#[doc(hidden)]
pub struct VasicekSampler<T: FloatExt> {
  ou: OuSampler<T>,
}

impl<T: FloatExt> PathSampler<T> for VasicekSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.ou.sample_into(out);
  }

  fn sample(&mut self) -> Array1<T> {
    self.ou.sample()
  }
}

py_process_1d!(PyVasicek, Vasicek,
  sig: (theta, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn sample_length_matches_n() {
    let v = Vasicek::<f64>::new(0.5, 0.04, 0.01, 100, Some(0.05), Some(1.0), Unseeded);
    let path = v.sample();
    assert_eq!(path.len(), 100);
  }

  #[test]
  fn sample_starts_at_x0() {
    let x0 = 0.05;
    let v = Vasicek::<f64>::new(0.5, 0.04, 0.01, 100, Some(x0), Some(1.0), Unseeded);
    let path = v.sample();
    assert!((path[0] - x0).abs() < 1e-12);
  }
}
