//! # SquaredBessel / Bessel
//!
//! The squared Bessel process of dimension δ, BESQ(δ):
//!
//! $$
//! dX_t=\delta\,dt+2\sqrt{|X_t|}\,dW_t
//! $$
//!
//! and its square root, the Bessel process of the same dimension, BES(δ):
//!
//! $$
//! dX_t=\frac{\delta-1}{2X_t}\,dt+dW_t
//! $$
//!
//! `Bessel` has the same law as `sqrt(SquaredBessel)` of the same dimension
//! δ (Revuz & Yor, *Continuous Martingales and Brownian Motion*, Ch. XI §1):
//! if `Z` solves the first SDE then `sqrt(Z)` solves the second, in law.
//! [`Cir`](crate::diffusion::cir::Cir) is, in turn, a time-changed and
//! scaled squared Bessel process: writing `Cir`'s own parameters (`theta` =
//! κ, `mu` = θ, `sigma` = σ), its path equals `e^{-κt} Z(τ(t))` for `Z`
//! a [`SquaredBessel`] of dimension `δ = 4κθ/σ²` under the time change
//! `τ(t) = (σ²/4κ)(e^{κt}-1)` — the two SDEs share the same `2√X` diffusion
//! shape, only reparametrized and time-changed.
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Squared Bessel process BESQ(δ).
///
/// `dX_t = delta * dt + 2 * sqrt(|X_t|) * dW_t`
///
/// See the module doc for the relationship to [`Bessel`] (its square root,
/// in law) and to [`Cir`](crate::diffusion::cir::Cir) (a time-changed,
/// scaled instance of this process).
pub struct SquaredBessel<T: FloatExt, S: SeedExt = Unseeded> {
  /// Dimension δ of the process. The literature convention is δ ≥ 0; δ ≥ 2
  /// additionally keeps the continuous-time process strictly positive once
  /// started away from 0 (Going-Jaeschke & Yor, 2003) — the direct analogue
  /// of `Cir`'s Feller condition, since `Cir` is itself a time-changed,
  /// scaled BESQ.
  pub delta: T,
  /// Number of points sampled along the BESQ path.
  pub n: usize,
  /// Initial value X₀ of the BESQ path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Enables reflect-about-zero variant when true; floors at zero
  /// otherwise (matching [`Cir::use_sym`](crate::diffusion::cir::Cir::use_sym)).
  pub use_sym: Option<bool>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> SquaredBessel<T, S> {
  /// Create a new SquaredBessel process.
  ///
  /// δ ≥ 2 keeps the continuous-time process strictly positive once started
  /// away from 0 — the direct analogue of [`Cir::new`](crate::diffusion::cir::Cir::new)'s
  /// Feller condition. Parameters violating it are accepted rather than
  /// rejected: the discretized step floors at zero by default, or reflects
  /// about zero when [`use_sym`](Self::use_sym) is `true`. A violation not
  /// paired with `use_sym = Some(true)` unconditionally prints a one-line
  /// diagnostic to stderr — including in release builds — and never panics
  /// (matching `Cir::new`).
  pub fn new(
    delta: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
    seed: S,
  ) -> Self {
    if delta < T::from_usize_(2) && use_sym != Some(true) {
      eprintln!(
        "warning: SquaredBessel::new: dimension below the strict-positivity \
         threshold (delta < 2) without use_sym = Some(true); the path floors \
         at zero on every boundary hit instead of reflecting — pass \
         use_sym = Some(true) for the standard sub-boundary mitigation"
      );
    }

    Self {
      delta,
      n,
      x0,
      t,
      use_sym,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for SquaredBessel<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = SquaredBesselSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> SquaredBesselSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    SquaredBesselSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      delta: self.delta,
      diff_scale: T::from_usize_(2),
      use_sym: self.use_sym.unwrap_or(false),
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`SquaredBessel`] sampling state: precomputed Euler scale and the
/// owned Gaussian source.
#[doc(hidden)]
pub struct SquaredBesselSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  delta: T,
  diff_scale: T,
  use_sym: bool,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> SquaredBesselSampler<T> {
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
      let dbesq = self.delta * self.dt + self.diff_scale * prev.abs().sqrt() * *z;
      let next = match self.use_sym {
        true => (prev + dbesq).abs(),
        false => (prev + dbesq).max(T::zero()),
      };
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for SquaredBesselSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("SquaredBessel output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PySquaredBessel, SquaredBessel,
  sig: (delta, n, x0=None, t=None, use_sym=None, seed=None, dtype=None),
  params: (delta: f64, n: usize, x0: Option<f64>, t: Option<f64>, use_sym: Option<bool>)
);

/// Bessel process BES(δ).
///
/// `dX_t = ((delta - 1) / (2 * X_t)) * dt + dW_t`
///
/// See the module doc: `Bessel` has the same law as `sqrt(`[`SquaredBessel`]`)`
/// of the same dimension δ, and [`Cir`](crate::diffusion::cir::Cir) is a
/// time-changed, scaled squared Bessel process.
///
/// The sampler below does not discretize the SDE above directly: its drift
/// `(delta-1)/(2X)` is singular at `X = 0`, and a plain Euler step that ever
/// floors to exactly 0 takes a division-by-near-zero drift kick on the very
/// next step — for `n = 200`, `t = 1` that single mishandled step is already
/// an ~5e9 excursion, large enough to dominate a terminal-mean Monte Carlo
/// estimate outright. Instead it runs the same, singularity-free BESQ(δ)
/// recursion [`SquaredBesselSampler`] uses internally (`2√X` vanishes
/// smoothly at the boundary, unlike `1/X`) and reports its square root — the
/// standard way this process is simulated in practice.
///
/// This sidesteps the singularity, but it is worth being precise about what
/// "exact" means here. The *law* identity `Bessel = sqrt(SquaredBessel)`
/// above (Revuz & Yor) is exact for the true continuous-time SDEs at every
/// `t`, which is why sampling BESQ and taking its square root is the right
/// thing to do at all. The *discretization* itself is not bias-free: it is
/// Euler-Maruyama on the BESQ recursion, which carries BESQ's own
/// O(dt)-class discretization bias, further reshaped (not removed) by the
/// nonlinear square root. This is a consistent, convergent approximation —
/// the bias shrinks toward 0 as `n → ∞` — not an exact transition kernel the
/// way, say, a noncentral-χ² exact CIR sampler would be. Worth keeping in
/// mind when choosing step counts for a calibration.
pub struct Bessel<T: FloatExt, S: SeedExt = Unseeded> {
  /// Dimension δ (see [`SquaredBessel::delta`] — the same δ ≥ 2 threshold
  /// keeps the process strictly positive).
  pub delta: T,
  /// Number of points sampled along the BES path.
  pub n: usize,
  /// Initial value X₀ of the BES path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Enables reflect-about-zero variant when true; floors at zero
  /// otherwise (see [`SquaredBessel::use_sym`]).
  pub use_sym: Option<bool>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Bessel<T, S> {
  /// Create a new Bessel process.
  ///
  /// Same δ ≥ 2 strict-positivity threshold and the same unconditional
  /// stderr diagnostic (never a panic) as
  /// [`SquaredBessel::new`] — see there for the full rationale.
  pub fn new(
    delta: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
    seed: S,
  ) -> Self {
    if delta < T::from_usize_(2) && use_sym != Some(true) {
      eprintln!(
        "warning: Bessel::new: dimension below the strict-positivity \
         threshold (delta < 2) without use_sym = Some(true); the path floors \
         at zero on every boundary hit instead of reflecting — pass \
         use_sym = Some(true) for the standard sub-boundary mitigation"
      );
    }

    Self {
      delta,
      n,
      x0,
      t,
      use_sym,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Bessel<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = BesselSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> BesselSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let x0 = self.x0.unwrap_or(T::zero());
    BesselSampler {
      n: self.n,
      x0,
      dt,
      delta: self.delta,
      diff_scale: T::from_usize_(2),
      use_sym: self.use_sym.unwrap_or(false),
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`Bessel`] sampling state: precomputed BESQ-recursion scale and
/// the owned Gaussian source (see [`Bessel`]'s doc for why the step is taken
/// in squared/BESQ space rather than on the Bessel SDE's singular drift
/// directly).
#[doc(hidden)]
pub struct BesselSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  delta: T,
  diff_scale: T,
  use_sym: bool,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> BesselSampler<T> {
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
    // Step the underlying BESQ(delta) state (see the struct doc) and report
    // its square root; `prev_sq` tracks X_t^2, never X_t itself, so the
    // `(delta-1)/(2X)` singularity never enters the recursion.
    let mut prev_sq = self.x0 * self.x0;
    for z in tail.iter_mut() {
      let dbesq = self.delta * self.dt + self.diff_scale * prev_sq.abs().sqrt() * *z;
      let next_sq = match self.use_sym {
        true => (prev_sq + dbesq).abs(),
        false => (prev_sq + dbesq).max(T::zero()),
      };
      *z = next_sq.sqrt();
      prev_sq = next_sq;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for BesselSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("Bessel output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyBessel, Bessel,
  sig: (delta, n, x0=None, t=None, use_sym=None, seed=None, dtype=None),
  params: (delta: f64, n: usize, x0: Option<f64>, t: Option<f64>, use_sym: Option<bool>)
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// E[X_t] = x0 + δ·t for BESQ(δ) (Revuz & Yor, *Continuous Martingales and
  /// Brownian Motion*, Ch. XI §1) — Monte Carlo check against the closed form.
  #[test]
  fn besq_mean_matches_closed_form() {
    let delta = 3.0;
    let x0 = 1.0;
    let t = 1.0;
    let n = 200;
    let paths = 20_000;
    let expected = x0 + delta * t;

    let best_rel_err = [2718u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let besq =
          SquaredBessel::<f64, _>::new(delta, n, Some(x0), Some(t), None, Deterministic::new(seed));
        let mean = besq
          .sample_par(paths)
          .iter()
          .map(|path| *path.last().unwrap())
          .sum::<f64>()
          / paths as f64;
        (mean - expected).abs() / expected
      })
      .fold(f64::INFINITY, f64::min);

    assert!(
      best_rel_err <= 2e-2,
      "best-of-3 relative error {best_rel_err} exceeds 2e-2 (expected {expected})"
    );
  }

  /// δ=1 is sub-boundary (δ < 2): the discretized path must stay
  /// non-negative and finite under both `use_sym` branches.
  #[test]
  fn besq_stays_nonnegative() {
    for use_sym in [None, Some(true)] {
      let besq = SquaredBessel::<f64, _>::new(
        1.0,
        500,
        Some(0.5),
        Some(1.0),
        use_sym,
        Deterministic::new(2718),
      );
      let path = besq.sample();
      assert!(
        path.iter().all(|x| x.is_finite() && *x >= 0.0),
        "use_sym = {use_sym:?}"
      );
    }
  }

  /// BES(δ) squared has the same law as BESQ(δ): compare terminal means.
  #[test]
  fn bessel_squared_matches_besq_mean() {
    let delta = 3.0;
    let x0 = 1.0;
    let t = 1.0;
    let n = 200;
    let paths = 20_000;

    let best_rel_err = [2718u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let besq =
          SquaredBessel::<f64, _>::new(delta, n, Some(x0), Some(t), None, Deterministic::new(seed));
        let besq_mean = besq
          .sample_par(paths)
          .iter()
          .map(|path| *path.last().unwrap())
          .sum::<f64>()
          / paths as f64;

        let bes = Bessel::<f64, _>::new(
          delta,
          n,
          Some(x0.sqrt()),
          Some(t),
          None,
          Deterministic::new(seed),
        );
        let bes_squared_mean = bes
          .sample_par(paths)
          .iter()
          .map(|path| path.last().unwrap().powi(2))
          .sum::<f64>()
          / paths as f64;

        (bes_squared_mean - besq_mean).abs() / besq_mean
      })
      .fold(f64::INFINITY, f64::min);

    assert!(
      best_rel_err <= 5e-2,
      "best-of-3 relative error {best_rel_err} exceeds 5e-2"
    );
  }
}
