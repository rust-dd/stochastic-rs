//! # CirPlusPlus (Cir++)
//!
//! $$
//! r_t=x_t+\varphi(t),\qquad dx_t=\kappa(\theta-x_t)\,dt+\sigma\sqrt{x_t}\,dW_t
//! $$
//!
//! Fields follow the CIR literature: `kappa` is the speed and `theta` the level, the reverse of [`Cir`]'s `theta`/`mu` (see the [glossary](crate::glossary)).
//! [`sampler`](ProcessExt::sampler) runs a [`Cir`] through its [`CirSampler`], so at `phi ≡ 0` the two agree bit for bit.
//!
//! Brigo, Mercurio (2006), Interest Rate Models, 2nd ed., §3.9, DOI 10.1007/978-3-540-34604-3.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::cir::Cir;
use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::diffusion::cir::CirSampler;
use crate::params::VarianceBoundary;
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
pub struct CirPlusPlus<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
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
  /// Treatment of a negative square-root state after a step; the Feller condition is not required.
  pub boundary: VarianceBoundary,
  /// Seed strategy (compile-time: [`Unseeded`] or
  /// the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Constant shift φ(t) ≡ 0 used by [`CirPlusPlus`]'s `Default` impl —
/// matches this file's own `zero_phi` test fixture. `φ ≡ 0` degenerates
/// `CirPlusPlus` to plain [`Cir`] exactly (see
/// `cir_pp_zero_shift_equals_cir` below), the most rigorously-tested
/// instance to default to.
fn default_phi<T: FloatExt>(_t: T) -> T {
  T::zero()
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `CirPlusPlus::default().with_kappa(3.0)`. No persisted cache:
/// `sampler()` builds a transient `Cir` (and its sampler) fresh from
/// `self`'s own fields every call.
impl<T: FloatExt, S: SeedExt> CirPlusPlus<T, S> {
  /// [`Cir::new`]'s Feller contract on the factor `x_t`: a violation under
  /// [`VarianceBoundary::Floor`] emits a `log::warn!`, never a panic.
  pub fn new(
    kappa: T,
    theta: T,
    sigma: T,
    phi: impl Into<Fn1D<T>>,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    boundary: VarianceBoundary,
    seed: S,
  ) -> Self {
    assert!(
      kappa.is_finite(),
      "kappa must satisfy `kappa.is_finite()`, got kappa = {kappa:?}"
    );
    assert!(
      theta.is_finite(),
      "theta must satisfy `theta.is_finite()`, got theta = {theta:?}"
    );
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      x0.is_none_or(|v| v.is_finite()),
      "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = {x0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    if T::from_usize_(2) * kappa * theta < sigma.powi(2) && boundary != VarianceBoundary::Reflect {
      log::warn!(
        "CirPlusPlus::new: Feller condition violated (2*kappa*theta < sigma^2) \
         under VarianceBoundary::Floor; the underlying CIR factor floors at zero on every \
         boundary hit instead of reflecting — pass VarianceBoundary::Reflect for the \
         standard sub-Feller mitigation"
      );
    }

    Self {
      backend: Cpu,
      kappa,
      theta,
      sigma,
      phi: phi.into(),
      n,
      x0,
      t,
      boundary,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> CirPlusPlus<T, S, B> {
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

  /// Replace `boundary`, all else unchanged.
  pub fn with_boundary(mut self, boundary: VarianceBoundary) -> Self {
    self.boundary = boundary;
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
/// [`Cir`]'s own `Default` parameterization
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
      VarianceBoundary::Floor,
      Unseeded,
    )
  }
}

/// The Euler engine's view of CIR++: the square-root recursion the wrapped
/// process runs, with the deterministic shift added in the report rather
/// than to the state, exactly as the host adds it.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for CirPlusPlus<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    match self.boundary {
      VarianceBoundary::Floor => crate::euler::EulerSpec::ShiftedSquareRoot {
        theta: self.kappa,
        mu: self.theta,
        sigma: self.sigma,
      },
      VarianceBoundary::Reflect => crate::euler::EulerSpec::ShiftedSquareRootMirrored {
        theta: self.kappa,
        mu: self.theta,
        sigma: self.sigma,
      },
    }
  }

  fn initial_value(&self) -> T {
    self.x0.unwrap_or(T::zero())
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  /// The shift at each grid point.
  fn curve(&self) -> Option<Vec<T>> {
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n.saturating_sub(1).max(1));
    Some(
      (0..self.n)
        .map(|i| self.phi.call(T::from_usize_(i) * dt))
        .collect(),
    )
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] CirPlusPlus<T, S> { kappa, theta, sigma, phi, n, x0, t, boundary, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for CirPlusPlus<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for CirPlusPlus<T, S, B>
{
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
  /// replaying (see this file's own git history), and
  /// reordering it would trade this asymmetry for that regression.
  fn advance_chunk_seed(&self) {
    self.seed.next_seed();
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
    let cir: Cir<T, S> = Cir {
      theta: self.kappa,
      mu: self.theta,
      sigma: self.sigma,
      n: self.n,
      x0: self.x0,
      t: self.t,
      boundary: self.boundary,
      seed: self.seed.clone(),
      backend: Cpu,
    };
    CirPlusPlusSampler {
      n: self.n,
      dt,
      phi: &self.phi,
      cir: cir.sampler(),
      scratch: Array1::<T>::zeros(self.n),
    }
  }

  /// Through the Euler engine: on a device the recursion runs in the kernel
  /// with its time-varying coefficient bound per step, on the host devices it
  /// is this process's own sampler, chunked exactly as `ProcessExt` chunks.
  fn sample(&self) -> Array1<T> {
    self.backend.euler_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    self.backend.euler_paths_map(self, m, f)
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    self.backend.euler_paths_map_view(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    self.backend.euler_paths(self, m)
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    self.backend.try_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    self.backend.try_euler_paths(self, m)
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

impl<T: FloatExt> crate::traits::Sealed for CirPlusPlusSampler<'_, T> {}

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
mod python;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyCirPlusPlus;

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
      VarianceBoundary::Floor,
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
      VarianceBoundary::Floor,
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
      VarianceBoundary::Floor,
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
      VarianceBoundary::Floor,
      Deterministic::new(seed),
    );

    let cir_path = cir.sample();
    let pp_path = cir_pp.sample();

    for (c, p) in cir_path.iter().zip(pp_path.iter()) {
      assert!((p - (c + 0.03)).abs() < 1e-12, "cir={c} cir_pp={p}");
    }
  }

  /// Sub-Feller parameters build and sample under reflection, as for `Cir::new`.
  #[test]
  fn cir_pp_accepts_sub_feller_under_reflection() {
    let cir_pp = CirPlusPlus::<f64, _>::new(
      0.5,
      0.1,
      1.0,
      zero_phi as fn(f64) -> f64,
      256,
      Some(0.1),
      Some(1.0),
      VarianceBoundary::Reflect,
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
      VarianceBoundary::Floor,
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
      VarianceBoundary::Floor,
      Deterministic::new(42),
    )
    .sample();
    assert_eq!(cir_pp1, cir_pp2);
  }
}
