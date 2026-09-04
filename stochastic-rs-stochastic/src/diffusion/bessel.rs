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
use crate::device::Cpu;
use crate::device::HostBackend;
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
#[derive(Clone)]
pub struct SquaredBessel<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
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
  /// Seed strategy (compile-time: [`Unseeded`] or
  /// the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on) or [`on_device`](Self::on_device).
  pub backend: B,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `SquaredBessel::default().with_delta(4.0)`. No persisted cache:
/// `sampler()` builds its Gaussian source fresh from `self` every call.
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
      backend: Cpu,
      delta,
      n,
      x0,
      t,
      use_sym,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> SquaredBessel<T, S, B> {
  /// Replace `delta`, all else unchanged.
  pub fn with_delta(mut self, delta: T) -> Self {
    self.delta = delta;
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

/// δ=3, x₀=1, t=1 — matches this file's own
/// `besq_mean_matches_closed_form` / `bessel_squared_matches_besq_mean` test
/// fixture (which itself runs at n=200, not the n=252 below); δ≥2 keeps the
/// process strictly positive. n=252 — one trading year of daily steps
/// (this crate's `Default` convention).
impl<T: FloatExt> Default for SquaredBessel<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(3.0),
      252,
      Some(T::one()),
      Some(T::one()),
      None,
      Unseeded,
    )
  }
}

backend_switch!([T: FloatExt, S: SeedExt] SquaredBessel<T, S> { delta, n, x0, t, use_sym, seed } via host);

impl<T: FloatExt, S: SeedExt, B: HostBackend> ProcessExt<T> for SquaredBessel<T, S, B> {
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
#[derive(Clone)]
pub struct Bessel<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
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
  /// Seed strategy (compile-time: [`Unseeded`] or
  /// the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on) or [`on_device`](Self::on_device).
  pub backend: B,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Bessel::default().with_delta(4.0)`. No persisted cache: `sampler()`
/// builds its Gaussian source fresh from `self` every call.
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
      backend: Cpu,
      delta,
      n,
      x0,
      t,
      use_sym,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Bessel<T, S, B> {
  /// Replace `delta`, all else unchanged.
  pub fn with_delta(mut self, delta: T) -> Self {
    self.delta = delta;
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

/// δ=3, x₀=1, t=1 — same δ/x₀/t as [`SquaredBessel`]'s `Default` (this
/// file's own `bessel_squared_matches_besq_mean` test fixture compares the
/// two directly, at n=200 there); δ≥2 keeps the process strictly positive.
/// n=252 — one trading year of daily steps (this crate's `Default`
/// convention).
impl<T: FloatExt> Default for Bessel<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(3.0),
      252,
      Some(T::one()),
      Some(T::one()),
      None,
      Unseeded,
    )
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Bessel<T, S> { delta, n, x0, t, use_sym, seed } via host);

impl<T: FloatExt, S: SeedExt, B: HostBackend> ProcessExt<T> for Bessel<T, S, B> {
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

// Split out to keep this file under the project's 600-line cap (both
// `SquaredBessel` and `Bessel` now carry full `with_*` builder setter
// surfaces on top of the models themselves). Same pattern as
// `volatility/bates_svj.rs`/`jump/bates.rs` from the previous wave.
#[cfg(test)]
#[path = "bessel_tests.rs"]
mod tests;
