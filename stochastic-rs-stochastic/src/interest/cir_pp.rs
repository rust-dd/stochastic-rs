//! # CirPlusPlus (Cir++)
//!
//! $$
//! r_t=x_t+\varphi(t),\qquad dx_t=\kappa(\theta-x_t)\,dt+\sigma\sqrt{x_t}\,dW_t
//! $$
//!
//! Shift-extended CIR short-rate model (Brigo & Mercurio, *Interest Rate
//! Models — Theory and Practice*, 2nd ed., 2006, §3.9): the CIR factor
//! `x_t` carries all of the stochastic dynamics, and the deterministic
//! shift `phi(t)` is fitted separately so that `r_t` matches today's
//! observed term structure exactly — the same shift-extension idea
//! [`Cir2F`](crate::interest::cir_2f::Cir2F) already applies to a
//! two-factor sum `x_t + y_t`.
//!
//! **Naming note.** Unlike [`Cir`], whose Rust field `theta` names the
//! mean-reversion *speed* (κ) and `mu` the long-run *level* (θ) — the
//! workspace's own θ=speed/μ=level convention, see the
//! [glossary](crate::glossary) — `CirPlusPlus` follows the CIR
//! literature's own κ/θ notation directly: [`kappa`](CirPlusPlus::kappa) is
//! the speed and [`theta`](CirPlusPlus::theta) is the long-run level. This is a
//! deliberate departure from `Cir`'s field names, not an inconsistency to
//! "fix": the shift-extension literature that motivates this type (Brigo
//! & Mercurio) itself writes the factor's SDE as `dx_t = κ(θ-x_t)dt +
//! ...`. See the glossary's θ (theta) table for this meaning alongside
//! [`BlackKarasinski::theta`](crate::interest::black_karasinski::BlackKarasinski::theta)'s
//! unrelated additive-drift-function meaning.
//!
//! Reuses [`Cir`]'s own sampler rather than re-deriving the discretization:
//! [`sampler`](ProcessExt::sampler) builds a plain [`Cir`] from this
//! struct's own `kappa`/`theta`/`sigma`/`n`/`x0`/`t`/`use_sym`/`seed`
//! fields (cloning the seed source — cheap and side-effect-free, see
//! [`SeedExt`]'s `Clone` supertrait) and keeps its real [`CirSampler`], so
//! at `phi ≡ 0` the two processes consume the same Gaussian stream through
//! the same step formula and agree bit-for-bit.
use ndarray::Array1;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::cir::Cir;
use crate::buffer::array1_from_fill;
use crate::diffusion::cir::CirSampler;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Cox-Ingersoll-Ross process with a deterministic shift (Cir++).
///
/// `r_t = x_t + phi(t)`, where `x_t` follows
/// `dx_t = kappa * (theta - x_t) * dt + sigma * sqrt(x_t) * dW_t`.
///
/// See the module doc for the `kappa`/`theta` naming note and the
/// composition with [`Cir`].
#[derive(Clone)]
pub struct CirPlusPlus<T: FloatExt, S: SeedExt = Unseeded> {
  /// Mean-reversion speed κ of the underlying CIR factor `x_t` (see the
  /// module doc: unlike [`Cir::theta`], the speed here is named `kappa`).
  pub kappa: T,
  /// Long-run mean level θ of the underlying CIR factor `x_t` (see the
  /// module doc: unlike [`Cir::mu`], the level here is named `theta`).
  pub theta: T,
  /// Diffusion scale σ multiplying `sqrt(x_t) dW_t`.
  pub sigma: T,
  /// Deterministic shift φ(t) fitting today's term structure, added to
  /// the CIR factor to form the observed short rate `r_t = x_t + phi(t)`.
  pub phi: Fn1D<T>,
  /// Number of points sampled along the path.
  pub n: usize,
  /// Initial value `x_0` of the CIR factor — not `r_0` directly; the
  /// observed short rate at `t=0` is `x_0 + phi(0)`.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Enables the symmetric/truncated update variant for the underlying
  /// CIR factor when true (see [`Cir::use_sym`]).
  pub use_sym: Option<bool>,
  /// Seed strategy (compile-time: [`Unseeded`] or
  /// [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
}

/// Constant shift φ(t) ≡ 0 used by [`CirPlusPlus`]'s `Default` impl —
/// matches this file's own `zero_phi` test fixture. `φ ≡ 0` degenerates
/// `CirPlusPlus` to plain [`Cir`] exactly (see
/// `cir_pp_zero_shift_equals_cir` below), the most rigorously-tested
/// instance to default to.
fn default_phi<T: FloatExt>(_t: T) -> T {
  T::zero()
}

impl<T: FloatExt, S: SeedExt> CirPlusPlus<T, S> {
  /// Create a new CirPlusPlus process.
  ///
  /// The Feller condition and its sub-Feller mitigation apply to the
  /// underlying CIR factor `x_t` exactly as in [`Cir::new`]: a violation
  /// not paired with `use_sym = Some(true)` unconditionally prints a
  /// one-line diagnostic to stderr — including in release builds — and
  /// never panics.
  pub fn new(
    kappa: T,
    theta: T,
    sigma: T,
    phi: impl Into<Fn1D<T>>,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
    seed: S,
  ) -> Self {
    if T::from_usize_(2) * kappa * theta < sigma.powi(2) && use_sym != Some(true) {
      eprintln!(
        "warning: CirPlusPlus::new: Feller condition violated (2*kappa*theta < sigma^2) \
         without use_sym = Some(true); the underlying CIR factor floors at zero on every \
         boundary hit instead of reflecting — pass use_sym = Some(true) for the standard \
         sub-Feller mitigation"
      );
    }

    Self {
      kappa,
      theta,
      sigma,
      phi: phi.into(),
      n,
      x0,
      t,
      use_sym,
      seed,
    }
  }

  /// Every field has a matching `with_*` builder setter, e.g.
  /// `CirPlusPlus::default().with_kappa(3.0)`. No persisted cache:
  /// `sampler()` builds a transient `Cir` (and its sampler) fresh from
  /// `self`'s own fields every call.
  /// Replace `kappa`, all else unchanged.
  pub fn with_kappa(mut self, kappa: T) -> Self {
    self.kappa = kappa;
    self
  }

  /// Replace `theta`, all else unchanged.
  pub fn with_theta(mut self, theta: T) -> Self {
    self.theta = theta;
    self
  }

  /// Replace `sigma`, all else unchanged.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    self.sigma = sigma;
    self
  }

  /// Replace the deterministic shift `phi`, all else unchanged.
  pub fn with_phi(mut self, phi: impl Into<Fn1D<T>>) -> Self {
    self.phi = phi.into();
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

/// κ=2.5, θ=0.04, σ=0.2, φ(t)≡0, x₀=0.04 — reuses
/// [`Cir`](crate::diffusion::cir::Cir)'s own `Default` parameterization
/// verbatim, since φ≡0 degenerates `CirPlusPlus` to `Cir` exactly (this
/// file's own `cir_pp_zero_shift_equals_cir` test proves it bit-for-bit).
/// Feller condition `2κθ = 0.2 ≥ σ² = 0.04` holds with a comfortable
/// margin. This deliberately does **not** reuse this file's own
/// `cir_pp_is_deterministic` test values (κ=0.5, θ=0.04, σ=0.2, x₀=0.03):
/// those sit exactly *on* the Feller boundary in real arithmetic
/// (`2·0.5·0.04 = 0.04 = 0.2²`), but `0.2 * 0.2 == 0.040000000000000001`
/// in `f64` — one ulp above `0.04` — so `CirPlusPlus::new`'s sub-Feller
/// guard fires and prints a warning on every construction. A `Default`
/// that warns on construction defeats the point of a zero-friction entry
/// point, so this picks a parameterization with real headroom instead.
/// t=1, n=252 — one trading year of daily steps (this crate's `Default`
/// convention).
impl<T: FloatExt> Default for CirPlusPlus<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(2.5),
      T::from_f64_fast(0.04),
      T::from_f64_fast(0.2),
      default_phi::<T> as fn(T) -> T,
      252,
      Some(T::from_f64_fast(0.04)),
      Some(T::one()),
      None,
      Unseeded,
    )
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for CirPlusPlus<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = CirPlusPlusSampler<'s, T>
  where
    Self: 's;

  /// `sampler()` clones `self.seed` into a transient `Cir` (a non-advancing
  /// snapshot), so each chunk's clone must see a distinct state.
  ///
  /// `CirPlusPlus` is the crate's one remaining user of this override (every
  /// other type that needed it before was rewritten to derive its own basis
  /// instead — see [`ProcessExt`]'s trait-level reproducibility section),
  /// which surfaces one known, undocumented-until-now asymmetry: this method
  /// runs *before* `sampler()` when called from
  /// [`chunked_samplers`](ProcessExt::chunked_samplers) (so `sample_par(m)`'s
  /// chunks each see a freshly-ticked state), but *after* sampling when
  /// called from [`sample`](ProcessExt::sample)'s default (so a fresh
  /// object's very first `.sample()` sees `self.seed`'s *un*-ticked starting
  /// state). A brand-new `CirPlusPlus`'s first `.sample()` and its first
  /// `sample_par(m)` chunk therefore consume different bases one tick apart.
  /// Not fixed here: the after-sampling placement in `sample`'s default was
  /// deliberately chosen so repeated `.sample()` calls advance instead of
  /// replaying (see this file's own git history / MIGRATION.md), and
  /// reordering it would trade this asymmetry for that regression.
  fn advance_chunk_seed(&self) {
    self.seed.seed_value();
  }

  fn sampler(&self) -> CirPlusPlusSampler<'_, T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    // A transient `Cir` built from this process's own fields, so the
    // shift extension reuses `Cir`'s actual sampler/discretization rather
    // than a hand-copied formula. Cloning `seed` is free of side effects
    // (`Deterministic::clone` snapshots the current state without
    // advancing it), so under the same seed this produces the identical
    // Gaussian stream a standalone `Cir` with the same parameters would.
    let cir = Cir {
      theta: self.kappa,
      mu: self.theta,
      sigma: self.sigma,
      n: self.n,
      x0: self.x0,
      t: self.t,
      use_sym: self.use_sym,
      seed: self.seed.clone(),
    };
    CirPlusPlusSampler {
      n: self.n,
      dt,
      phi: &self.phi,
      cir: cir.sampler(),
      scratch: Array1::<T>::zeros(self.n),
    }
  }
}

/// Reusable [`CirPlusPlus`] sampling state: the real [`CirSampler`] for the
/// underlying factor `x_t`, a scratch buffer to receive it, and the
/// borrowed shift function `phi` added on top.
#[doc(hidden)]
pub struct CirPlusPlusSampler<'a, T: FloatExt> {
  n: usize,
  dt: T,
  phi: &'a Fn1D<T>,
  cir: CirSampler<T>,
  scratch: Array1<T>,
}

impl<T: FloatExt> CirPlusPlusSampler<'_, T> {
  fn fill_path(&mut self, out: &mut [T]) {
    self.cir.sample_into(&mut self.scratch);
    for (i, (dst, &x)) in out.iter_mut().zip(self.scratch.iter()).enumerate() {
      *dst = x + self.phi.call(T::from_usize_(i) * self.dt);
    }
  }
}

impl<T: FloatExt> PathSampler<T> for CirPlusPlusSampler<'_, T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("CirPlusPlus output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyCirPlusPlus {
  inner: Option<CirPlusPlus<f64>>,
  seeded: Option<CirPlusPlus<f64, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyCirPlusPlus {
  #[new]
  #[pyo3(signature = (kappa, theta, sigma, phi, n, x0=None, t=None, use_sym=None, seed=None))]
  fn new(
    kappa: f64,
    theta: f64,
    sigma: f64,
    phi: pyo3::Py<pyo3::PyAny>,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    use_sym: Option<bool>,
    seed: Option<u64>,
  ) -> Self {
    match seed {
      Some(s) => Self {
        inner: None,
        seeded: Some(CirPlusPlus::new(
          kappa,
          theta,
          sigma,
          Fn1D::Py(phi),
          n,
          x0,
          t,
          use_sym,
          Deterministic::new(s),
        )),
      },
      None => Self {
        inner: Some(CirPlusPlus::new(
          kappa,
          theta,
          sigma,
          Fn1D::Py(phi),
          n,
          x0,
          t,
          use_sym,
          Unseeded,
        )),
        seeded: None,
      },
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch_f64!(self, |inner| inner
      .sample()
      .into_pyarray(py)
      .into_py_any(py)
      .unwrap())
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  fn zero_phi(_t: f64) -> f64 {
    0.0
  }

  fn const_phi(_t: f64) -> f64 {
    0.03
  }

  /// φ ≡ 0 must reproduce the plain CIR path bit-for-bit under the same seed.
  #[test]
  fn cir_pp_zero_shift_equals_cir() {
    let kappa = 0.5;
    let theta = 0.04;
    let sigma = 0.2;
    let x0 = 0.03;
    let n = 128;
    let t = 1.0;
    let seed = 2718u64;

    let cir = Cir::<f64, _>::new(
      kappa,
      theta,
      sigma,
      n,
      Some(x0),
      Some(t),
      None,
      Deterministic::new(seed),
    );
    let cir_pp = CirPlusPlus::<f64, _>::new(
      kappa,
      theta,
      sigma,
      zero_phi as fn(f64) -> f64,
      n,
      Some(x0),
      Some(t),
      None,
      Deterministic::new(seed),
    );

    let cir_path = cir.sample();
    let pp_path = cir_pp.sample();

    assert_eq!(cir_path.len(), pp_path.len());
    for (c, p) in cir_path.iter().zip(pp_path.iter()) {
      assert_eq!(c.to_bits(), p.to_bits(), "cir={c} cir_pp={p}");
    }
  }

  /// r_t = x_t + φ(t): a constant shift must translate every path point exactly.
  #[test]
  fn cir_pp_constant_shift_translates_path() {
    let kappa = 0.5;
    let theta = 0.04;
    let sigma = 0.2;
    let x0 = 0.03;
    let n = 128;
    let t = 1.0;
    let seed = 999u64;

    let cir = Cir::<f64, _>::new(
      kappa,
      theta,
      sigma,
      n,
      Some(x0),
      Some(t),
      None,
      Deterministic::new(seed),
    );
    let cir_pp = CirPlusPlus::<f64, _>::new(
      kappa,
      theta,
      sigma,
      const_phi as fn(f64) -> f64,
      n,
      Some(x0),
      Some(t),
      None,
      Deterministic::new(seed),
    );

    let cir_path = cir.sample();
    let pp_path = cir_pp.sample();

    for (c, p) in cir_path.iter().zip(pp_path.iter()) {
      assert!((p - (c + 0.03)).abs() < 1e-12, "cir={c} cir_pp={p}");
    }
  }

  /// Sub-Feller parameters must be accepted (not panic) with `use_sym =
  /// Some(true)`, exactly like `Cir::new`'s own precedent.
  #[test]
  fn cir_pp_accepts_sub_feller_with_use_sym() {
    let cir_pp = CirPlusPlus::<f64, _>::new(
      0.5,
      0.1,
      1.0,
      zero_phi as fn(f64) -> f64,
      256,
      Some(0.1),
      Some(1.0),
      Some(true),
      Deterministic::new(7),
    );
    let path = cir_pp.sample();
    assert_eq!(path.len(), 256);
    assert!(path.iter().all(|x| x.is_finite()));
  }

  /// Same seed twice must be bit-identical.
  #[test]
  fn cir_pp_is_deterministic() {
    let cir_pp1 = CirPlusPlus::<f64, _>::new(
      0.5,
      0.04,
      0.2,
      zero_phi as fn(f64) -> f64,
      100,
      Some(0.03),
      Some(1.0),
      None,
      Deterministic::new(42),
    )
    .sample();
    let cir_pp2 = CirPlusPlus::<f64, _>::new(
      0.5,
      0.04,
      0.2,
      zero_phi as fn(f64) -> f64,
      100,
      Some(0.03),
      Some(1.0),
      None,
      Deterministic::new(42),
    )
    .sample();
    assert_eq!(cir_pp1, cir_pp2);
  }
}
