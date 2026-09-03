//! Teng Modified Ou stochastic correlation (Eq. 19/20, Lemma 1).
//!
//! $$
//! dX_t = \kappa(\mu - \tanh(X_t))\,dt + \sigma\,dW_t, \quad \rho_t = \tanh(X_t)
//! $$
//!
//! The SCP ρ_t = tanh(X_t) satisfies:
//!
//! $$
//! d\rho_t = (1-\rho_t^2)\bigl(\kappa(\mu-\rho_t) - \sigma^2\rho_t\bigr)\,dt
//!         + (1-\rho_t^2)\sigma\,dW_t
//! $$
//!
//! Diffusion vanishes *quadratically* at ±1 (stronger confinement).
//! Closed-form stationary density: f(ρ̃) ∝ (1+ρ̃)^{a+b}(1−ρ̃)^{a−b}.

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

/// Teng modified-Ou stochastic correlation process.
///
/// Simulates in X-space via the modified Ou (Eq. 19):
/// dX_t = κ(μ − tanh(X_t)) dt + σ dW_t
///
/// Output ρ_t = tanh(X_t) ∈ (−1, 1).
#[derive(Debug, Clone)]
pub struct TengSCP<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Mean-reversion speed (κ > 0).
  pub kappa: T,
  /// Long-run correlation level (μ ∈ (−1, 1)).
  pub mu: T,
  /// Correlation volatility (σ > 0). Not validated by [`TengSCP::new`]: at
  /// `sigma = 0`, [`density_a`](TengSCP::density_a)/
  /// [`density_b`](TengSCP::density_b)/
  /// [`stationary_density_unnorm`](TengSCP::stationary_density_unnorm) all
  /// divide by `sigma * sigma`, so those (but not path simulation itself,
  /// which never divides by `sigma`) return non-finite values — see
  /// [`stationary_density_unnorm`](TengSCP::stationary_density_unnorm)'s own
  /// doc for the exact breakdown.
  pub sigma: T,
  /// Initial correlation (ρ₀ ∈ (−1, 1)).
  pub rho0: T,
  /// Number of points sampled along the correlation path.
  pub n: usize,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy.
  pub seed: S,
  /// Sampling backend marker (compile-time): [`Cpu`] by default, a device
  /// marker after [`on`](Self::on). Public so `..Default::default()` struct
  /// updates keep working; it carries no data.
  pub backend: PhantomData<B>,
}

impl<T: FloatExt, S: SeedExt> TengSCP<T, S> {
  pub fn new(kappa: T, mu: T, sigma: T, rho0: T, n: usize, t: Option<T>, seed: S) -> Self {
    Self {
      backend: PhantomData,
      kappa,
      mu,
      sigma,
      rho0,
      n,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> TengSCP<T, S, B> {}

impl<T: FloatExt, S: SeedExt, B> TengSCP<T, S, B> {
  /// Reparametrised coefficients (Eq. 21):
  /// κ* = κ + σ², μ* = κμ/(κ+σ²), σ* = σ.
  pub fn kappa_star(&self) -> T {
    self.kappa + self.sigma * self.sigma
  }

  pub fn mu_star(&self) -> T {
    self.kappa * self.mu / self.kappa_star()
  }

  /// Stationary density exponents (Eq. 33/37):
  /// a = (κ − σ²)/σ², b = κμ/σ².
  ///
  /// At `sigma = 0` this divides by zero: `+inf` for `kappa > 0` (its own
  /// required domain), not `NaN` — a positive value divided by exact zero
  /// is a well-defined signed infinity in IEEE 754.
  pub fn density_a(&self) -> T {
    let s2 = self.sigma * self.sigma;
    (self.kappa - s2) / s2
  }

  /// At `sigma = 0`: `NaN` if `mu = 0` (a literal `0/0`), otherwise a
  /// signed infinity matching `mu`'s sign — see [`density_a`](Self::density_a)'s
  /// own doc for why a nonzero numerator over zero is infinite, not `NaN`.
  pub fn density_b(&self) -> T {
    let s2 = self.sigma * self.sigma;
    self.kappa * self.mu / s2
  }

  /// Evaluate the (unnormalised) stationary density at ρ̃ ∈ (−1, 1).
  ///
  /// f(ρ̃) ∝ (1+ρ̃)^{a+b} (1−ρ̃)^{a−b}   (Eq. 39)
  ///
  /// At `sigma = 0` this is `NaN` for *every* `mu`, even though `density_a`
  /// and `density_b` individually are merely infinite for `mu != 0`: `a` is
  /// always `+inf` there, and `a + b` / `a - b` hits `inf + (-inf)` or
  /// `inf - inf` for one of the two combinations regardless of `b`'s sign,
  /// which is `NaN` even though neither `a` nor `b` alone was.
  pub fn stationary_density_unnorm(&self, rho: T) -> T {
    let a = self.density_a();
    let b = self.density_b();
    if rho <= -T::one() || rho >= T::one() {
      return T::zero();
    }
    let log_f = (a + b) * (T::one() + rho).ln() + (a - b) * (T::one() - rho).ln();
    log_f.exp()
  }

  /// Effective parameters κ*, μ*, σ* (van Emmerich form).
  pub fn effective_params(&self) -> (T, T, T) {
    (self.kappa_star(), self.mu_star(), self.sigma)
  }
}

backend_switch!([T: FloatExt, S: SeedExt] TengSCP<T, S> { kappa, mu, sigma, rho0, n, t, seed } via host);

impl<T: FloatExt, S: SeedExt, B: HostBackend> ProcessExt<T> for TengSCP<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = TengSCPSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> TengSCPSampler<T> {
    let n_steps = self.n.saturating_sub(1);
    let dt = if n_steps > 0 {
      self.t.unwrap_or(T::one()) / T::from_usize_(n_steps)
    } else {
      T::zero()
    };
    TengSCPSampler {
      n: self.n,
      kappa: self.kappa,
      mu: self.mu,
      sigma: self.sigma,
      rho0: self.rho0,
      dt,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`TengSCP`] sampling state: owns the Gaussian source and the
/// precomputed step size. `fill_path` Euler-steps the modified Ou in X-space
/// (`dX = κ(μ − tanh X)dt + σ dW`) and outputs `ρ = tanh(X)` in place; the owned
/// source advances each call for independent paths.
#[doc(hidden)]
pub struct TengSCPSampler<T: FloatExt> {
  n: usize,
  kappa: T,
  mu: T,
  sigma: T,
  rho0: T,
  dt: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> TengSCPSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    let n_steps = out.len() - 1;
    let mut gn = Array1::<T>::zeros(n_steps);
    if let Some(slice) = gn.as_slice_mut() {
      self.normal.fill_slice(slice);
    }

    let x0 = self
      .rho0
      .clamp(T::from_f64_fast(-0.999), T::from_f64_fast(0.999))
      .atanh();
    let mut x = x0;
    out[0] = x.tanh();

    for i in 1..out.len() {
      // Modified Ou: dX = κ(μ - tanh(X))dt + σ dW
      let drift = self.kappa * (self.mu - x.tanh());
      x = x + drift * self.dt + self.sigma * gn[i - 1];
      out[i] = x.tanh();
    }
  }
}

impl<T: FloatExt> PathSampler<T> for TengSCPSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("TengSCP output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyTengSCP, TengSCP,
  sig: (kappa, mu, sigma, rho0, n, t=None, seed=None, dtype=None),
  params: (kappa: f64, mu: f64, sigma: f64, rho0: f64, n: usize, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  #[test]
  fn stays_bounded() {
    let scp = TengSCP::new(
      8.0_f64,
      0.0,
      1.2,
      0.3,
      2000,
      Some(2.0),
      Deterministic::new(123),
    );
    let path = scp.sample();
    assert!(path.iter().all(|&r| r > -1.0 && r < 1.0));
  }

  #[test]
  fn mean_reverts() {
    let mu = -0.4_f64;
    let scp = TengSCP::new(12.0, mu, 0.5, 0.5, 5000, Some(10.0), Deterministic::new(99));
    let path = scp.sample();
    let tail = &path.as_slice().unwrap()[4000..];
    let avg: f64 = tail.iter().sum::<f64>() / tail.len() as f64;
    assert!(
      (avg - mu).abs() < 0.15,
      "Expected mean near {mu}, got {avg}"
    );
  }

  #[test]
  fn stationary_density_peaks_near_mu() {
    let scp = TengSCP::new(8.0_f64, 0.3, 0.5, 0.0, 100, None, Unseeded);
    let d_at_mu = scp.stationary_density_unnorm(0.3);
    let d_at_0 = scp.stationary_density_unnorm(0.0);
    assert!(d_at_mu > d_at_0);
  }

  /// Backs the doc comments on `sigma`/`density_a`/`density_b`/
  /// `stationary_density_unnorm`: `TengSCP::new` does not validate that
  /// `sigma` is positive, so this must actually reach the documented
  /// non-finite outputs rather than just claim to.
  #[test]
  fn zero_sigma_density_functions_are_non_finite() {
    let zero_mu = TengSCP::new(8.0_f64, 0.0, 0.0, 0.0, 10, Some(1.0), Unseeded);
    assert_eq!(zero_mu.density_a(), f64::INFINITY);
    assert!(zero_mu.density_b().is_nan(), "mu=0 makes density_b a 0/0");
    assert!(zero_mu.stationary_density_unnorm(0.0).is_nan());

    let positive_mu = TengSCP::new(8.0_f64, 0.3, 0.0, 0.0, 10, Some(1.0), Unseeded);
    assert_eq!(positive_mu.density_a(), f64::INFINITY);
    assert_eq!(positive_mu.density_b(), f64::INFINITY);
    assert!(
      positive_mu.stationary_density_unnorm(0.0).is_nan(),
      "a - b is inf - inf even though a and b were each merely infinite"
    );

    let negative_mu = TengSCP::new(8.0_f64, -0.3, 0.0, 0.0, 10, Some(1.0), Unseeded);
    assert_eq!(negative_mu.density_b(), f64::NEG_INFINITY);
    assert!(
      negative_mu.stationary_density_unnorm(0.0).is_nan(),
      "a + b is inf + (-inf) this time, but still NaN"
    );
  }

  #[test]
  fn seeded_reproducibility() {
    let p1 = TengSCP::new(
      5.0_f64,
      0.0,
      0.8,
      0.0,
      200,
      Some(1.0),
      Deterministic::new(42),
    )
    .sample();
    let p2 = TengSCP::new(
      5.0_f64,
      0.0,
      0.8,
      0.0,
      200,
      Some(1.0),
      Deterministic::new(42),
    )
    .sample();
    for i in 0..200 {
      assert!((p1[i] - p2[i]).abs() < 1e-14, "diverged at i={i}");
    }
  }
}
