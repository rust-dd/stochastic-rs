//! # fCIR
//!
//! $$
//! dX_t=\kappa(\theta-X_t)dt+\sigma\sqrt{X_t}\,dB_t^H
//! $$
//!
//! Reference: Mishura Y., Yurchenko-Tytarenko A. (2018) — *Fractional
//! Cox-Ingersoll-Ross Process with Non-Zero "Mean"*, Modern Stochastics:
//! Theory and Applications 5(1), 99–111, DOI: 10.15559/18-vmsta97 — the
//! non-zero-mean-reverting fCIR process this file discretises by Euler
//! scheme, floored or reflected at zero like
//! [`Cir`](crate::diffusion::cir::Cir).
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::buffer::array1_from_fill;
use crate::device::Backend;
use crate::device::Cpu;
use crate::noise::fgn::Fgn;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Fractional Cox-Ingersoll-Ross (Fcir) process.
/// dX(t) = theta(mu - X(t))dt + sigma * sqrt(X(t))dW^H(t)
/// where X(t) is the Fcir process.
pub struct Fcir<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Mean-reversion speed (κ in the module header). Multiplies
  /// `(mu - X_t)`, despite the field's own name.
  pub theta: T,
  /// Long-run mean level (θ in the module header). The level `X`
  /// reverts to between fractional-noise shocks.
  pub mu: T,
  /// Diffusion scale σ multiplying `√X_t dB_t^H`.
  pub sigma: T,
  /// Number of points sampled along the fCIR path.
  pub n: usize,
  /// Initial value X₀ of the fCIR path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Enables symmetric/truncated update variant when true.
  pub use_sym: Option<bool>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  fgn: Fgn<T, Unseeded, B>,
}

impl<T: FloatExt, S: SeedExt> Fcir<T, S, Cpu> {
  /// Create a new Fcir process.
  ///
  /// Same Feller-condition contract as [`crate::diffusion::cir::Cir`]:
  /// `2·theta·mu ≥ sigma²` keeps the continuous-time process strictly
  /// positive, but sub-Feller parameters are accepted rather than
  /// rejected, since the discretised step already keeps every sample
  /// non-negative — floored at zero by default, or reflected when
  /// [`use_sym`](Self::use_sym) is `true`. A violation not paired with
  /// `use_sym = Some(true)` unconditionally prints a one-line diagnostic
  /// to stderr — including in release builds; it never panics.
  #[must_use]
  pub fn new(
    hurst: T,
    theta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
    seed: S,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");
    if T::from_usize_(2) * theta * mu < sigma.powi(2) && use_sym != Some(true) {
      eprintln!(
        "warning: Fcir::new: Feller condition violated (2*theta*mu < sigma^2) \
         without use_sym = Some(true); the path floors at zero on every \
         boundary hit instead of reflecting — pass use_sym = Some(true) for \
         the standard sub-Feller mitigation"
      );
    }

    Self {
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      use_sym,
      seed,
      fgn: Fgn::new(hurst, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt, B: Backend> ProcessExt<T> for Fcir<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = FcirSampler<'s, T, S, B>
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
  fn sampler(&self) -> FcirSampler<'_, T, S, B> {
    FcirSampler {
      fcir: self,
      seed: self.seed.derive(),
    }
  }
}

/// Reusable [`Fcir`] sampling state: borrows the process for its inner [`Fgn`]
/// and owns a seed derived once at construction. The path is an Euler
/// discretisation of `dX = theta(mu - X) dt + sigma sqrt(X) dB^H`, clamped at
/// zero (or reflected when `use_sym`) so the variance stays non-negative.
#[doc(hidden)]
pub struct FcirSampler<'a, T: FloatExt, S: SeedExt, B> {
  fcir: &'a Fcir<T, S, B>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt, B: Backend> FcirSampler<'_, T, S, B> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    let p = self.fcir;
    let dt = p.fgn.dt();
    let fgn = p.fgn.noise(&self.seed);
    let use_sym = p.use_sym.unwrap_or(false);

    out[0] = p.x0.unwrap_or(T::zero());
    let mut prev = out[0];
    for (dst, inc) in out[1..].iter_mut().zip(fgn.iter()) {
      let dfcir = p.theta * (p.mu - prev) * dt + p.sigma * prev.abs().sqrt() * *inc;
      let next = match use_sym {
        true => (prev + dfcir).abs(),
        false => (prev + dfcir).max(T::zero()),
      };
      *dst = next;
      prev = next;
    }
  }
}

impl<T: FloatExt, S: SeedExt, B: Backend> PathSampler<T> for FcirSampler<'_, T, S, B> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Fcir output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.fcir.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Fcir<T, S> { hurst, theta, mu, sigma, n, x0, t, use_sym, seed } via fgn);

py_process_1d!(PyFcir, Fcir,
  sig: (hurst, theta, mu, sigma, n, x0=None, t=None, use_sym=None, seed=None, dtype=None),
  params: (hurst: f64, theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, use_sym: Option<bool>)
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// `2*theta*mu = 2*0.5*0.1 = 0.1 < sigma^2 = 1.0` — Feller condition
  /// violated, mirroring the equivalent `Cir` test. `use_sym = Some(true)`
  /// must build and sample without panicking.
  #[test]
  fn fcir_accepts_sub_feller_with_use_sym() {
    let fcir = Fcir::<f64, _>::new(
      0.7,
      0.5,
      0.1,
      1.0,
      256,
      Some(0.1),
      Some(1.0),
      Some(true),
      Deterministic::new(7),
    );
    let path = fcir.sample();
    assert_eq!(path.len(), 256);
    assert!(
      path.iter().all(|x| x.is_finite()),
      "sub-Feller Fcir path must stay finite under use_sym = Some(true)"
    );
  }

  /// The default (floor-at-zero) scheme must also accept sub-Feller
  /// parameters without panicking — only the diagnostic warning differs.
  #[test]
  fn fcir_accepts_sub_feller_without_use_sym() {
    let fcir = Fcir::<f64, _>::new(
      0.7,
      0.5,
      0.1,
      1.0,
      256,
      Some(0.1),
      Some(1.0),
      None,
      Deterministic::new(7),
    );
    let path = fcir.sample();
    assert!(path.iter().all(|x| x.is_finite() && *x >= 0.0));
  }
}
