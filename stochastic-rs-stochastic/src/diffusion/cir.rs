//! # Cir
//!
//! $$
//! dX_t=\kappa(\theta-X_t)\,dt+\sigma\sqrt{X_t}\,dW_t
//! $$
//!
//! Reference: Cox J. C., Ingersoll J. E., Ross S. A. (1985) — *A Theory
//! of the Term Structure of Interest Rates*, Econometrica 53(2),
//! 385–407, DOI: 10.2307/1911242.
//!
use std::marker::PhantomData;

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::euler::EulerBackend;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;
use crate::traits::process::sample_map_chunked;
use crate::traits::process::sample_par_chunked;

/// Cox-Ingersoll-Ross (Cir) process.
///
/// `dX(t) = theta * (mu - X(t)) * dt + sigma * sqrt(X(t)) * dW(t)`
///
/// In the SDE notation `dX = κ(θ − X) dt + σ √X dW` the Rust field
/// [`theta`](Self::theta) corresponds to κ (mean-reversion speed) and
/// [`mu`](Self::mu) corresponds to θ (long-run mean level).
#[derive(Clone)]
pub struct Cir<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Mean-reversion speed (κ in the SDE). Controls how fast `X` is pulled
  /// back toward [`mu`](Self::mu).
  pub theta: T,
  /// Long-run mean level (θ in the SDE). The value `X` reverts to as
  /// `t → ∞`.
  pub mu: T,
  /// Diffusion scale σ multiplying `√X_t dW_t` (σ in the SDE).
  pub sigma: T,
  /// Number of points sampled along the CIR path.
  pub n: usize,
  /// Initial value X₀ of the CIR path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Enables symmetric/truncated update variant when true.
  pub use_sym: Option<bool>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// Sampling backend marker (compile-time): [`Cpu`] by default, a device
  /// marker after [`on`](Self::on). Public so `..Default::default()` struct
  /// updates keep working; it carries no data.
  pub backend: PhantomData<B>,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Cir::default().with_theta(3.0).with_sigma(0.3)`. No persisted cache:
/// `sampler()` builds its Gaussian stream fresh from `self` every call.
impl<T: FloatExt, S: SeedExt> Cir<T, S> {
  /// Create a new Cir process.
  ///
  /// The Feller condition `2·theta·mu ≥ sigma²` keeps the *continuous-time*
  /// process strictly positive. Parameters that violate it are accepted
  /// rather than rejected: the discretised step already keeps every
  /// sample non-negative regardless — floored at zero by default, or
  /// reflected about zero when [`use_sym`](Self::use_sym) is `true`,
  /// which is the documented way to handle sub-Feller paths (matching the
  /// same-shaped variance factor in
  /// [`Heston`](crate::volatility::heston::Heston), which imposes no
  /// Feller precondition at all). A violation not paired with
  /// `use_sym = Some(true)` unconditionally prints a one-line diagnostic
  /// to stderr — including in release builds, where real Monte Carlo /
  /// calibration runs happen and silently biased boundary handling is
  /// exactly what a caller needs to know about; it never panics.
  pub fn new(
    theta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
    seed: S,
  ) -> Self {
    if T::from_usize_(2) * theta * mu < sigma.powi(2) && use_sym != Some(true) {
      eprintln!(
        "warning: Cir::new: Feller condition violated (2*theta*mu < sigma^2) \
         without use_sym = Some(true); the path floors at zero on every \
         boundary hit instead of reflecting — pass use_sym = Some(true) for \
         the standard sub-Feller mitigation"
      );
    }

    Self {
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      use_sym,
      seed,
      backend: PhantomData,
    }
  }

  /// Replace `theta`, all else unchanged.
  pub fn with_theta(mut self, theta: T) -> Self {
    self.theta = theta;
    self
  }

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

  /// Replace `x0`, all else unchanged.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    self.x0 = x0;
    self
  }

  /// Replace `use_sym`, all else unchanged.
  pub fn with_use_sym(mut self, use_sym: Option<bool>) -> Self {
    self.use_sym = use_sym;
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

/// θ=2.5, μ=0.04, σ=0.2, x₀=0.04, use_sym=false — a textbook Cir
/// parameterization; Feller condition `2θμ = 0.2 ≥ σ² = 0.04` holds. t=1,
/// n=252 — one trading year of daily steps (this crate's `Default`
/// convention).
impl<T: FloatExt> Default for Cir<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(2.5),
      T::from_f64_fast(0.04),
      T::from_f64_fast(0.2),
      252,
      Some(T::from_f64_fast(0.04)),
      Some(T::one()),
      Some(false),
      Unseeded,
    )
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Cir<T, S> { theta, mu, sigma, n, x0, t, use_sym, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: EulerBackend<T>> ProcessExt<T> for Cir<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = CirSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> CirSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    CirSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      theta: self.theta,
      mu: self.mu,
      diff_scale: self.sigma,
      use_sym: self.use_sym.unwrap_or(false),
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }

  fn sample(&self) -> Array1<T> {
    if B::DEVICE {
      B::euler_paths(self, 1).remove(0)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    if B::DEVICE {
      B::euler_paths(self, m).iter().map(f).collect()
    } else {
      sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    if B::DEVICE {
      B::euler_paths(self, m)
    } else {
      sample_par_chunked(self, m)
    }
  }
}

/// Reusable [`Cir`] sampling state.
#[doc(hidden)]
pub struct CirSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  theta: T,
  mu: T,
  diff_scale: T,
  use_sym: bool,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> CirSampler<T> {
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
    let mut prev = self.x0;
    for z in tail.iter_mut() {
      let dcir = self.theta * (self.mu - prev) * self.dt + self.diff_scale * prev.abs().sqrt() * *z;
      let next = match self.use_sym {
        true => (prev + dcir).abs(),
        false => (prev + dcir).max(T::zero()),
      };
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for CirSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Cir output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyCir, Cir,
  sig: (theta, mu, sigma, n, x0=None, t=None, use_sym=None, seed=None, dtype=None),
  params: (theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, use_sym: Option<bool>)
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// `2*theta*mu = 2*0.5*0.1 = 0.1 < sigma^2 = 1.0` — Feller condition
  /// violated. `use_sym = Some(true)` is the documented mitigation:
  /// construction and sampling must succeed without panicking, and the
  /// path must stay finite.
  #[test]
  fn cir_accepts_sub_feller_with_use_sym() {
    let cir = Cir::<f64, _>::new(
      0.5,
      0.1,
      1.0,
      256,
      Some(0.1),
      Some(1.0),
      Some(true),
      Deterministic::new(7),
    );
    let path = cir.sample();
    assert_eq!(path.len(), 256);
    assert!(
      path.iter().all(|x| x.is_finite()),
      "sub-Feller Cir path must stay finite under use_sym = Some(true)"
    );
  }

  /// The default (floor-at-zero) scheme must also accept sub-Feller
  /// parameters without panicking — only the diagnostic warning differs.
  #[test]
  fn cir_accepts_sub_feller_without_use_sym() {
    let cir = Cir::<f64, _>::new(
      0.5,
      0.1,
      1.0,
      256,
      Some(0.1),
      Some(1.0),
      None,
      Deterministic::new(7),
    );
    let path = cir.sample();
    assert!(path.iter().all(|x| x.is_finite() && *x >= 0.0));
  }
}
