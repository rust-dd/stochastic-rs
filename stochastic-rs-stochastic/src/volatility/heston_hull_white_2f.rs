//! # Heston Hull-White 2F
//!
//! $$
//! \begin{aligned}
//! d\log S_t&=\bigl(r_t-\tfrac12 v_t\bigr)dt+\sqrt{v_t}\,dW^x_t\\
//! dv_t&=\kappa(\theta-v_t)\,dt+\sigma\sqrt{v_t}\,dW^v_t\\
//! dr_t&=\bigl[\theta_r(t)+u_t-a\,r_t\bigr]dt+\sigma_1\,dW^r_t\\
//! du_t&=-b\,u_t\,dt+\sigma_2\,dW^u_t,\qquad d\langle W^i,W^j\rangle_t=\rho_{ij}\,dt
//! \end{aligned}
//! $$
//!
//! The Heston-Gaussian two-factor hybrid (H-G2++): Heston equity on the [`HullWhite2F`] short rate, the
//! four drivers `(x, v, r, u)` under one correlation matrix, simulated in full rather than as the affine
//! approximation the paper prices with. The output is [`HestonHullWhite2FPaths`].
//!
//! The rate enters the spot and the integrated short rate through the same trapezoidal sum, so the
//! discounted spot carries no discretisation bias from the rate leg; the variance step is [`Euler`]
//! (full truncation) or [`AndersenQe`], the latter with the rate-variance correlations at zero as in
//! the paper's matrix (2.4).
//!
//! Reference: Grzelak L. A., Oosterlee C. W. & van Weeren S. (2011), *The affine Heston model with
//! correlated Gaussian interest rates for pricing hybrid derivatives*, Quantitative Finance 11(11),
//! 1647-1663, DOI: 10.1080/14697688.2011.615216.
//!
//! [`HullWhite2F`]: crate::interest::hull_white_2f::HullWhite2F

use std::marker::PhantomData;

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::linalg::validate_correlation;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

mod scheme;
pub use scheme::HybridScheme;

pub use crate::volatility::heston::AndersenQe;
pub use crate::volatility::heston::Euler;

/// Spot, variance, short rate, auxiliary rate factor and the integrated short rate `∫₀ᵗ r`.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct HestonHullWhite2FPaths<T> {
  pub s: Array1<T>,
  pub v: Array1<T>,
  pub r: Array1<T>,
  pub u: Array1<T>,
  pub integrated_rate: Array1<T>,
}

impl<T: FloatExt> HestonHullWhite2FPaths<T> {
  fn zeros(n: usize) -> Self {
    let row = || Array1::<T>::zeros(n);
    Self {
      s: row(),
      v: row(),
      r: row(),
      u: row(),
      integrated_rate: row(),
    }
  }

  fn assert_points(&self, n: usize) {
    let rows = [&self.s, &self.v, &self.r, &self.u, &self.integrated_rate];
    assert!(
      rows.iter().all(|row| row.len() == n),
      "every output row must hold n = {n} points"
    );
  }
}

#[derive(Debug, Clone)]
pub struct HestonHullWhite2F<T: FloatExt, Sch: HybridScheme = Euler, S: SeedExt = Unseeded, B = Cpu>
{
  /// Initial spot.
  pub s0: T,
  /// Initial variance, not a volatility.
  pub v0: T,
  /// Mean-reversion speed κ of the variance.
  pub kappa: T,
  /// Long-run variance θ, a variance and not a volatility.
  pub theta: T,
  /// Volatility of variance σ.
  pub sigma: T,
  /// Drift target θ_r(t) of the short rate, the additive curve term `HullWhite2F::theta` carries.
  pub theta_r: Fn1D<T>,
  /// Mean-reversion speed of the short rate.
  pub a: T,
  /// Volatility of the short rate.
  pub sigma1: T,
  /// Volatility of the auxiliary factor `u`.
  pub sigma2: T,
  /// Mean-reversion speed of `u`.
  pub b: T,
  /// Correlation matrix of the drivers in the order `(x, v, r, u)`: `4 × 4`, unit diagonal,
  /// symmetric and positive definite.
  pub rho: Array2<T>,
  /// Initial short rate (defaults to 0); `u` starts at 0.
  pub r0: Option<T>,
  /// Grid points including `t = 0`, at least 2.
  pub n: usize,
  /// Horizon (defaults to 1).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// Cholesky factor of `rho` in the driver order the scheme steps, see [`HybridScheme::factor`].
  chol: Array2<T>,
  _scheme: PhantomData<Sch>,
  /// The sampling backend: [`Cpu`] by default, a host handle after [`on`](Self::on).
  pub backend: B,
}

/// `rho` must be a `4 × 4` correlation matrix; the Cholesky factorisation rejects an indefinite one.
fn checked_factor<T: FloatExt, Sch: HybridScheme>(rho: &Array2<T>) -> Array2<T> {
  let dim = rho.dim();
  assert!(
    dim == (4, 4),
    "rho must satisfy `rho.dim() == (4, 4)`, got rho.dim() = {dim:?}"
  );
  validate_correlation(rho);
  Sch::factor(rho)
}

impl<T: FloatExt, S: SeedExt> HestonHullWhite2F<T, Euler, S> {
  /// Panics unless every parameter is finite, `s0 > 0`, `v0`, `kappa`, `theta`, `sigma`, `sigma1` and
  /// `sigma2` are non-negative, `n >= 2`, a given `t` is positive and `rho` is a `4 × 4` correlation.
  pub fn new(
    s0: T,
    v0: T,
    kappa: T,
    theta: T,
    sigma: T,
    theta_r: impl Into<Fn1D<T>>,
    a: T,
    sigma1: T,
    sigma2: T,
    b: T,
    rho: Array2<T>,
    r0: Option<T>,
    n: usize,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self::check_s0(s0);
    Self::check_v0(v0);
    Self::check_kappa(kappa);
    Self::check_theta(theta);
    Self::check_sigma(sigma);
    Self::check_a(a);
    Self::check_sigma1(sigma1);
    Self::check_sigma2(sigma2);
    Self::check_b(b);
    Self::check_r0(r0);
    check_n(n);
    check_t(t);
    let chol = checked_factor::<T, Euler>(&rho);
    Self {
      s0,
      v0,
      kappa,
      theta,
      sigma,
      theta_r: theta_r.into(),
      a,
      sigma1,
      sigma2,
      b,
      rho,
      r0,
      n,
      t,
      seed,
      chol,
      _scheme: PhantomData,
      backend: Cpu,
    }
  }
}

/// The drift target that holds the short rate's mean at the 3 percent of the paper's experiment 4.2.
fn flat_drift_target<T: FloatExt>(_t: T) -> T {
  T::from_f64_fast(1.1 * 0.03)
}

/// Grzelak, Oosterlee and van Weeren (2011), experiment 4.2: κ = 0.8, θ = 0.2, σ = 0.2, a = 1.1,
/// σ₁ = 0.01, b = 0.8, σ₂ = 0.015, ρ_xv = −0.3, ρ_xr = 0.35, ρ_xu = 0.08, ρ_ru = −0.4, S₀ = 1, r₀ = 0.03.
impl<T: FloatExt> Default for HestonHullWhite2F<T, Euler, Unseeded> {
  fn default() -> Self {
    let f = T::from_f64_fast;
    let rho = ndarray::array![
      [f(1.0), f(-0.3), f(0.35), f(0.08)],
      [f(-0.3), f(1.0), f(0.0), f(0.0)],
      [f(0.35), f(0.0), f(1.0), f(-0.4)],
      [f(0.08), f(0.0), f(-0.4), f(1.0)],
    ];
    Self::new(
      T::one(),
      f(0.2),
      f(0.8),
      f(0.2),
      f(0.2),
      flat_drift_target::<T> as fn(T) -> T,
      f(1.1),
      f(0.01),
      f(0.015),
      f(0.8),
      rho,
      Some(f(0.03)),
      252,
      Some(T::one()),
      Unseeded,
    )
  }
}

/// Every field has a matching `with_*` setter, checked as `new` checks it, chaining after [`qe`](Self::qe) too.
impl<T: FloatExt, Sch: HybridScheme, S: SeedExt, B> HestonHullWhite2F<T, Sch, S, B> {
  fn check_s0(s0: T) {
    require!(s0.is_finite(), s0);
    require!(s0 > T::zero(), s0);
  }

  fn check_v0(v0: T) {
    require!(v0.is_finite(), v0);
    require!(v0 >= T::zero(), v0);
  }

  fn check_kappa(kappa: T) {
    require!(kappa.is_finite(), kappa);
    require!(kappa >= T::zero(), kappa);
  }

  fn check_theta(theta: T) {
    require!(theta.is_finite(), theta);
    require!(theta >= T::zero(), theta);
  }

  fn check_sigma(sigma: T) {
    require!(sigma.is_finite(), sigma);
    require!(sigma >= T::zero(), sigma);
  }

  fn check_a(a: T) {
    require!(a.is_finite(), a);
  }

  fn check_sigma1(sigma1: T) {
    require!(sigma1.is_finite(), sigma1);
    require!(sigma1 >= T::zero(), sigma1);
  }

  fn check_sigma2(sigma2: T) {
    require!(sigma2.is_finite(), sigma2);
    require!(sigma2 >= T::zero(), sigma2);
  }

  fn check_b(b: T) {
    require!(b.is_finite(), b);
  }

  fn check_r0(r0: Option<T>) {
    require!(r0.is_none_or(|r0| r0.is_finite()), r0);
  }

  pub fn with_s0(mut self, s0: T) -> Self {
    Self::check_s0(s0);
    self.s0 = s0;
    self
  }

  pub fn with_v0(mut self, v0: T) -> Self {
    Self::check_v0(v0);
    self.v0 = v0;
    self
  }

  pub fn with_kappa(mut self, kappa: T) -> Self {
    Self::check_kappa(kappa);
    self.kappa = kappa;
    self
  }

  pub fn with_theta(mut self, theta: T) -> Self {
    Self::check_theta(theta);
    self.theta = theta;
    self
  }

  pub fn with_sigma(mut self, sigma: T) -> Self {
    Self::check_sigma(sigma);
    self.sigma = sigma;
    self
  }

  pub fn with_theta_r(mut self, theta_r: impl Into<Fn1D<T>>) -> Self {
    self.theta_r = theta_r.into();
    self
  }

  pub fn with_a(mut self, a: T) -> Self {
    Self::check_a(a);
    self.a = a;
    self
  }

  pub fn with_sigma1(mut self, sigma1: T) -> Self {
    Self::check_sigma1(sigma1);
    self.sigma1 = sigma1;
    self
  }

  pub fn with_sigma2(mut self, sigma2: T) -> Self {
    Self::check_sigma2(sigma2);
    self.sigma2 = sigma2;
    self
  }

  pub fn with_b(mut self, b: T) -> Self {
    Self::check_b(b);
    self.b = b;
    self
  }

  /// Replaces the correlation matrix and refactors it in this scheme's driver order.
  pub fn with_rho(mut self, rho: Array2<T>) -> Self {
    self.chol = checked_factor::<T, Sch>(&rho);
    self.rho = rho;
    self
  }

  pub fn with_r0(mut self, r0: Option<T>) -> Self {
    Self::check_r0(r0);
    self.r0 = r0;
    self
  }

  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self
  }

  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self
  }

  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }

  /// Time step `t / (n - 1)`.
  pub fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }

  /// QE's preconditions, checked at the switch and again where the scheme runs, since the fields stay public.
  fn assert_qe_parameters(&self) {
    let (kappa, theta, sigma) = (self.kappa, self.theta, self.sigma);
    require!(kappa > T::zero(), kappa);
    require!(theta > T::zero(), theta);
    require!(sigma > T::zero(), sigma);
    for (what, index) in [("rho[(1, 2)]", (1, 2)), ("rho[(1, 3)]", (1, 3))] {
      let value = self.rho[index];
      assert!(
        value == T::zero(),
        "the rate-variance correlation {what} must satisfy `{what} == T::zero()` under QE, got {what} = {value:?}"
      );
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> HestonHullWhite2F<T, Euler, S, B> {
  /// Re-tags the model with the [`AndersenQe`] variance step; panics unless `kappa`, `theta` and
  /// `sigma` are positive and the rate-variance correlations `rho[(1, 2)]`, `rho[(1, 3)]` are zero.
  pub fn qe(self) -> HestonHullWhite2F<T, AndersenQe, S, B> {
    let model = HestonHullWhite2F {
      s0: self.s0,
      v0: self.v0,
      kappa: self.kappa,
      theta: self.theta,
      sigma: self.sigma,
      theta_r: self.theta_r,
      a: self.a,
      sigma1: self.sigma1,
      sigma2: self.sigma2,
      b: self.b,
      chol: checked_factor::<T, AndersenQe>(&self.rho),
      rho: self.rho,
      r0: self.r0,
      n: self.n,
      t: self.t,
      seed: self.seed,
      _scheme: PhantomData,
      backend: self.backend,
    };
    model.assert_qe_parameters();
    model
  }
}

backend_switch!([T: FloatExt, Sch: HybridScheme, S: SeedExt] HestonHullWhite2F<T, Sch, S> { s0, v0, kappa, theta, sigma, theta_r, a, sigma1, sigma2, b, rho, r0, n, t, seed, chol, _scheme } via euler);

impl<T: FloatExt, S: SeedExt, Sch: HybridScheme, B: crate::euler::EulerBackend<T>>
  crate::traits::Sealed for HestonHullWhite2F<T, Sch, S, B>
{
}

impl<T: FloatExt, S: SeedExt, Sch: HybridScheme, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for HestonHullWhite2F<T, Sch, S, B>
{
  type Output = HestonHullWhite2FPaths<T>;
  type Sampler<'s>
    = HestonHullWhite2FSampler<'s, T, Sch, S, B>
  where
    Self: 's;

  /// Derives (not clones) `self.seed`, so every chunk of `sample_par` owns a hash-mixed basis.
  fn sampler(&self) -> HestonHullWhite2FSampler<'_, T, Sch, S, B> {
    HestonHullWhite2FSampler {
      model: self,
      seed: self.seed.derive(),
    }
  }

  /// No device family carries the hybrid yet, so every configuration samples on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    Some("no device family steps the hybrid's four drivers yet")
  }
}

/// Reusable [`HestonHullWhite2F`] sampler: borrows the model for `theta_r` and owns a seed derived
/// once, which the scheme's streams advance on every path of the chunk.
#[doc(hidden)]
#[derive(Debug)]
pub struct HestonHullWhite2FSampler<'a, T: FloatExt, Sch: HybridScheme, S: SeedExt, B> {
  model: &'a HestonHullWhite2F<T, Sch, S, B>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt, Sch: HybridScheme, B: Send + Sync> crate::traits::Sealed
  for HestonHullWhite2FSampler<'_, T, Sch, S, B>
{
}

impl<T: FloatExt, S: SeedExt, Sch: HybridScheme, B: Send + Sync> PathSampler<T>
  for HestonHullWhite2FSampler<'_, T, Sch, S, B>
{
  type Output = HestonHullWhite2FPaths<T>;

  fn sample_into(&mut self, out: &mut HestonHullWhite2FPaths<T>) {
    Sch::simulate_into(self.model, &self.seed, out);
  }

  fn sample(&mut self) -> HestonHullWhite2FPaths<T> {
    let mut out = HestonHullWhite2FPaths::zeros(self.model.n);
    self.sample_into(&mut out);
    out
  }
}

#[cfg(test)]
mod tests;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyHestonHullWhite2F;
