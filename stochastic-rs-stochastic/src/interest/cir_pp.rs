//! CIR++: a CIR factor plus a deterministic shift `phi`; `phi ≡ 0` is [`Cir`]'s path bit for bit.
//!
//! $$
//! r_t=x_t+\varphi(t),\qquad dx_t=\kappa(\theta-x_t)\,dt+\sigma\sqrt{x_t}\,dW_t
//! $$
//!
//! `kappa` is the speed and `theta` the level, where [`Cir`] names them `theta` and `mu`.
//!
//! Reference: Brigo & Mercurio (2006), 2nd ed., §3.9, DOI 10.1007/978-3-540-34604-3

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::diffusion::cir::Cir;
use crate::diffusion::cir::CirSampler;
use crate::params::VarianceBoundary;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// CIR with a deterministic shift: `r_t = x_t + phi(t)` on a [`Cir`] factor `x_t`, whose speed
/// and level are named `kappa` and `theta` here.
#[derive(Debug, Clone)]
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
  /// Initial value; `None` starts at zero.
  /// Of the CIR factor, so the short rate starts at `x0 + phi(0)`.
  pub x0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
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

/// The shift φ ≡ 0 of [`CirPlusPlus`]'s `Default`, at which the process is plain [`Cir`] bit
/// for bit (`cir_pp_zero_shift_equals_cir`).
fn default_phi<T: FloatExt>(_t: T) -> T {
  T::zero()
}

/// Every field has a `with_*` setter (`CirPlusPlus::default().with_kappa(3.0)`); nothing is
/// cached, since `sampler()` builds a transient `Cir` from `self` on every call.
impl<T: FloatExt, S: SeedExt> CirPlusPlus<T, S> {
  /// [`Cir::new`]'s Feller contract on `x_t`: a violation under [`VarianceBoundary::Floor`] logs a
  /// warning. Panics unless `n >= 2`, a given `t` is positive and every parameter is finite.
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
    Self::check_kappa(kappa);
    Self::check_theta(theta);
    Self::check_sigma(sigma);
    check_n(n);
    Self::check_x0(x0);
    check_t(t);
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
  fn check_kappa(kappa: T) {
    require!(kappa.is_finite(), kappa);
  }

  fn check_theta(theta: T) {
    require!(theta.is_finite(), theta);
  }

  fn check_sigma(sigma: T) {
    require!(sigma.is_finite(), sigma);
  }

  fn check_x0(x0: Option<T>) {
    require!(x0.is_none_or(|v| v.is_finite()), x0);
  }

  /// Replace `kappa`; panics unless `kappa` is finite.
  pub fn with_kappa(mut self, kappa: T) -> Self {
    Self::check_kappa(kappa);
    self.kappa = kappa;
    self
  }

  /// Replace `theta`; panics unless `theta` is finite.
  pub fn with_theta(mut self, theta: T) -> Self {
    Self::check_theta(theta);
    self.theta = theta;
    self
  }

  /// Replace `sigma`; panics unless `sigma` is finite.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    Self::check_sigma(sigma);
    self.sigma = sigma;
    self
  }

  /// Replace the deterministic shift `phi`, all else unchanged.
  pub fn with_phi(mut self, phi: impl Into<Fn1D<T>>) -> Self {
    self.phi = phi.into();
    self
  }

  /// Replace `x0`; panics unless `x0` is finite.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    Self::check_x0(x0);
    self.x0 = x0;
    self
  }

  /// Replace `boundary`, all else unchanged.
  pub fn with_boundary(mut self, boundary: VarianceBoundary) -> Self {
    self.boundary = boundary;
    self
  }

  /// Replace `n`, the grid points along the path; panics unless `n >= 2`.
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self
  }

  /// Replace the simulation horizon `t`; panics unless `t`, when given, is positive and finite.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// [`Cir`]'s `Default` (κ=2.5, θ=0.04, σ=0.2, x₀=0.04) at φ≡0, clear of the Feller boundary,
/// where `f64` rounding fires the warning; t=1, 252 points, a trading year of daily closes.
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

/// The Euler engine's view: the wrapped square-root recursion, with the deterministic shift
/// added in the report, not to the state, as the host adds it.
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

  /// `sampler()` clones `self.seed` into a transient `Cir`, so each chunk must see a new state.
  /// `chunked_samplers` ticks before sampling and `sample` after, so the two start one tick apart.
  fn advance_chunk_seed(&self) {
    self.seed.next_seed();
  }

  fn sampler(&self) -> CirPlusPlusSampler<'_, T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    // A transient `Cir` of this process's fields, so the shift reuses `Cir`'s own sampler. A
    // cloned seed does not advance, so the stream is a standalone `Cir`'s under the same seed.
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

  /// Through the Euler engine: a device binds the time-varying coefficient per step in its
  /// kernel, the host runs this process's own sampler, chunked as `ProcessExt` chunks.
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

/// Reusable [`CirPlusPlus`] sampling state: the [`CirSampler`] of the factor `x_t`, a scratch
/// buffer for it, and the borrowed shift `phi`.
#[doc(hidden)]
#[derive(Debug)]
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
