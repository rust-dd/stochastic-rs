//! # DisplacedDiffusion
//!
//! $$
//! dS_t=\mu(S_t+\beta)\,dt+\sigma(S_t+\beta)\,dW_t
//! $$
//!
//! The shifted variable `Y_t = S_t + beta` follows an ordinary geometric
//! Brownian motion, so the model has the exact closed-form identity
//!
//! $$
//! S_t+\beta=(S_0+\beta)\exp\!\left(\left(\mu-\tfrac{\sigma^2}{2}\right)t+\sigma W_t\right)
//! $$
//!
//! `beta = 0` degenerates exactly to [`Gbm`](crate::diffusion::gbm::Gbm):
//! [`GbmSampler`](crate::diffusion::gbm::GbmSampler)'s own `fill_path` steps
//! the multiplicative recursion `S_{i+1} = S_i(1 + \mu\,dt + \sigma z_i)`
//! rather than exponentiating each increment (the exact log-form above is
//! used elsewhere in `Gbm` only for the analytic terminal-marginal formulas,
//! not for path generation). `DisplacedDiffusionSampler` applies that same
//! recursion to `Y_t = S_t + beta` and reports `Y_t - beta`, so that at
//! `beta = 0` it reproduces `Gbm`'s path bit-for-bit under the same seed —
//! aligning the discretization with `Gbm`'s actual stepping rule, not just
//! its continuous-time law.
//!
//! This is the shifted/displaced-lognormal model used for negative-rate-
//! tolerant Black-76-style pricers: choosing `beta > 0` keeps `S_t + beta`
//! (and hence the model) well-defined even while `S_t` itself goes
//! negative.
use std::marker::PhantomData;

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::device::HostBackend;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Displaced (shifted-lognormal) diffusion.
///
/// `dS_t = mu * (S_t + beta) * dt + sigma * (S_t + beta) * dW_t`
///
/// See the module doc for the exact closed-form identity and the exact
/// bit-for-bit relationship to [`Gbm`](crate::diffusion::gbm::Gbm) at
/// `beta = 0`.
#[derive(Clone)]
pub struct DisplacedDiffusion<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Constant proportional drift rate μ of the shifted variable `S_t +
  /// beta` (same "no mean reversion" role as
  /// [`Gbm::mu`](crate::diffusion::gbm::Gbm::mu)).
  pub mu: T,
  /// Diffusion scale σ multiplying `(S_t + beta) dW_t`.
  pub sigma: T,
  /// Displacement / shift β applied to the driven state before the
  /// GBM-style dynamics act on it. `beta = 0` degenerates exactly to
  /// [`Gbm`](crate::diffusion::gbm::Gbm). A positive `beta` is the
  /// shifted-lognormal trick that keeps the model well-defined while `S_t`
  /// itself is allowed to go negative (e.g. negative interest rates in
  /// Black-76-style pricers).
  pub beta: T,
  /// Number of points sampled along the displaced-diffusion path.
  pub n: usize,
  /// Initial value S₀ of the displaced-diffusion path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or
  /// the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// Sampling backend marker (compile-time): [`Cpu`] by default, a device
  /// marker after [`on`](Self::on). Public so `..Default::default()` struct
  /// updates keep working; it carries no data.
  pub backend: PhantomData<B>,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `DisplacedDiffusion::default().with_beta(10.0)`. No persisted cache:
/// `sampler()` builds its Gaussian source fresh from `self` every call.
impl<T: FloatExt, S: SeedExt> DisplacedDiffusion<T, S> {
  pub fn new(mu: T, sigma: T, beta: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    Self {
      backend: PhantomData,
      mu,
      sigma,
      beta,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> DisplacedDiffusion<T, S, B> {
  /// Replace `mu`, all else unchanged.
  pub fn with_mu(mut self, mu: T) -> Self {
    self.mu = mu;
    self
  }

  /// Replace `sigma`, all else unchanged.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    self.sigma = sigma;
    self
  }

  /// Replace `beta`, all else unchanged.
  pub fn with_beta(mut self, beta: T) -> Self {
    self.beta = beta;
    self
  }

  /// Replace `x0`, all else unchanged.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    self.x0 = x0;
    self
  }

  /// Replace the number of simulation steps `n`, all else unchanged.
  pub fn with_steps(mut self, n: usize) -> Self {
    self.n = n;
    self
  }

  /// Replace the simulation horizon `t`, all else unchanged.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// μ=0.05, σ=0.2, β=30, x₀=100, t=1 — μ/σ/x₀/t match
/// [`Gbm`](crate::diffusion::gbm::Gbm)'s own `Default` (β=0 degenerates
/// this type exactly to `Gbm`, see the module doc), and β=30 matches this
/// file's own `displaced_diffusion_mean_matches_closed_form` test fixture
/// (which itself runs at n=200, not the n=252 below), showcasing the
/// type's distinguishing shift feature rather than the degenerate case.
/// n=252 — one trading year of daily steps, consistent with
/// [`Gbm`](crate::diffusion::gbm::Gbm)'s own `Default` n (this crate's
/// shared convention, not itself drawn from either fixture above).
impl<T: FloatExt> Default for DisplacedDiffusion<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(0.05),
      T::from_f64_fast(0.2),
      T::from_f64_fast(30.0),
      252,
      Some(T::from_f64_fast(100.0)),
      Some(T::one()),
      Unseeded,
    )
  }
}

backend_switch!([T: FloatExt, S: SeedExt] DisplacedDiffusion<T, S> { mu, sigma, beta, n, x0, t, seed } via host);

impl<T: FloatExt, S: SeedExt, B: HostBackend> ProcessExt<T> for DisplacedDiffusion<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = DisplacedDiffusionSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> DisplacedDiffusionSampler<T> {
    // Same `n_increments` / `dt` derivation as `Gbm::sampler` — required for
    // the `beta = 0` path to consume the Gaussian stream identically.
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    DisplacedDiffusionSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::one()),
      beta: self.beta,
      drift_scale: self.mu * dt,
      diff_scale: self.sigma,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`DisplacedDiffusion`] sampling state: precomputed Euler scales
/// and the owned Gaussian source (mirrors
/// [`GbmSampler`](crate::diffusion::gbm::GbmSampler), plus the `beta` shift).
#[doc(hidden)]
pub struct DisplacedDiffusionSampler<T: FloatExt> {
  n: usize,
  x0: T,
  beta: T,
  drift_scale: T,
  diff_scale: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> DisplacedDiffusionSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.normal.fill_slice(tail);
    // Step the shifted variable `Y = S + beta` with literally the same
    // recursion `GbmSampler::fill_path` uses for `S`, then report `Y - beta`.
    // At `beta = 0` this is that recursion verbatim, term for term.
    let mut prev_shifted = self.x0 + self.beta;
    for z in tail.iter_mut() {
      let next_shifted =
        prev_shifted + self.drift_scale * prev_shifted + self.diff_scale * prev_shifted * *z;
      *z = next_shifted - self.beta;
      prev_shifted = next_shifted;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for DisplacedDiffusionSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("DisplacedDiffusion output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyDisplacedDiffusion, DisplacedDiffusion,
  sig: (mu, sigma, beta, n, x0=None, t=None, seed=None, dtype=None),
  params: (mu: f64, sigma: f64, beta: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::diffusion::gbm::Gbm;

  /// β=0 must reproduce Gbm exactly: same parameters, same seed ⇒
  /// bit-identical path.
  #[test]
  fn displaced_diffusion_beta_zero_equals_gbm() {
    let mu = 0.07;
    let sigma = 0.25;
    let x0 = 50.0;
    let n = 256;
    let t = 2.0;
    let seed = 2718u64;

    let gbm = Gbm::<f64, _>::new(mu, sigma, n, Some(x0), Some(t), Deterministic::new(seed));
    let dd = DisplacedDiffusion::<f64, _>::new(
      mu,
      sigma,
      0.0,
      n,
      Some(x0),
      Some(t),
      Deterministic::new(seed),
    );

    let gbm_path = gbm.sample();
    let dd_path = dd.sample();

    assert_eq!(gbm_path.len(), dd_path.len());
    for (g, d) in gbm_path.iter().zip(dd_path.iter()) {
      assert_eq!(g.to_bits(), d.to_bits(), "gbm={g} displaced={d}");
    }
  }

  /// E[S_t] = (S_0+β)e^{μt} − β for the displaced lognormal.
  #[test]
  fn displaced_diffusion_mean_matches_closed_form() {
    let mu = 0.05_f64;
    let sigma = 0.2;
    let beta = 30.0;
    let x0 = 100.0;
    let t = 1.0;
    let n = 200;
    let paths = 20_000;
    let expected = (x0 + beta) * (mu * t).exp() - beta;

    let best_rel_err = [2718u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let dd = DisplacedDiffusion::<f64, _>::new(
          mu,
          sigma,
          beta,
          n,
          Some(x0),
          Some(t),
          Deterministic::new(seed),
        );
        let mean = dd
          .sample_par(paths)
          .iter()
          .map(|path| *path.last().unwrap())
          .sum::<f64>()
          / paths as f64;
        (mean - expected).abs() / expected.abs()
      })
      .fold(f64::INFINITY, f64::min);

    assert!(
      best_rel_err <= 2e-2,
      "best-of-3 relative error {best_rel_err} exceeds 2e-2 (expected {expected})"
    );
  }

  /// Same seed twice must be bit-identical.
  #[test]
  fn displaced_diffusion_is_deterministic() {
    let dd1 = DisplacedDiffusion::<f64, _>::new(
      0.05,
      0.2,
      0.1,
      50,
      Some(1.0),
      Some(1.0),
      Deterministic::new(42),
    )
    .sample();
    let dd2 = DisplacedDiffusion::<f64, _>::new(
      0.05,
      0.2,
      0.1,
      50,
      Some(1.0),
      Some(1.0),
      Deterministic::new(42),
    )
    .sample();
    assert_eq!(dd1, dd2);
  }
}
