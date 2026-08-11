//! # Cir
//!
//! $$
//! dX_t=\kappa(\theta-X_t)\,dt+\sigma\sqrt{X_t}\,dW_t
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Cox-Ingersoll-Ross (Cir) process.
///
/// `dX(t) = theta * (mu - X(t)) * dt + sigma * sqrt(X(t)) * dW(t)`
///
/// In the SDE notation `dX = κ(θ − X) dt + σ √X dW` the Rust field
/// [`theta`](Self::theta) corresponds to κ (mean-reversion speed) and
/// [`mu`](Self::mu) corresponds to θ (long-run mean level).
pub struct Cir<T: FloatExt, S: SeedExt = Unseeded> {
  /// Mean-reversion speed (κ in the SDE). Controls how fast `X` is pulled
  /// back toward [`mu`](Self::mu).
  pub theta: T,
  /// Long-run mean level (θ in the SDE). The value `X` reverts to as
  /// `t → ∞`.
  pub mu: T,
  /// Diffusion / noise scale parameter (σ in the SDE).
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Enables symmetric/truncated update variant when true.
  pub use_sym: Option<bool>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

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
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Cir<T, S> {
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
