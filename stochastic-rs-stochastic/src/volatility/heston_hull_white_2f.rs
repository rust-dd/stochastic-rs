//! Heston equity on two-factor Hull-White rates (H-G2++), simulated in full by `Euler` or `AndersenQe`.
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
//! The rate enters the spot and `∫r` through one trapezoidal sum, so the discounted spot carries
//! no discretisation bias from the rate leg.
//!
//! Reference: Grzelak et al. (2011), Quant. Finance 11(11), 1647-1663, DOI 10.1080/14697688.2011.615216

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
use crate::volatility::heston::AndersenQe;
use crate::volatility::heston::Euler;

mod scheme;
pub use scheme::HybridScheme;

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
  /// Mean-reversion speed of the variance.
  pub kappa: T,
  /// Long-run level of the variance, itself a variance.
  pub theta: T,
  /// Volatility of the variance.
  pub sigma: T,
  /// Drift target θ_r(t) of the short rate, as `HullWhite2F::theta`; a non-finite value makes the
  /// rate rows, the spot and the integrated rate NaN from that grid point on.
  pub theta_r: Fn1D<T>,
  /// Mean-reversion speed of `r`, which reverts to `(theta_r(t) + u) / a`.
  pub a: T,
  /// Volatility of `r`, per unit time and in rate units, not relative.
  pub sigma1: T,
  /// Volatility of the auxiliary factor `u`, which enters the drift of `r`.
  pub sigma2: T,
  /// Mean-reversion speed of `u` towards zero.
  pub b: T,
  /// Correlation of the drivers in the order `(x, v, r, u)`; [`with_rho`](Self::with_rho) replaces
  /// it together with its factor.
  rho: Array2<T>,
  /// Initial short rate (defaults to 0); `u` starts at 0.
  pub r0: Option<T>,
  /// Grid points including `t = 0`, at least 2.
  pub n: usize,
  /// Horizon (defaults to 1).
  pub t: Option<T>,
  /// [`Unseeded`] or a [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// Cholesky factor of `rho` in the driver order the scheme steps, see [`HybridScheme::factor`].
  chol: Array2<T>,
  _scheme: PhantomData<Sch>,
  /// The sampling backend: [`Cpu`] by default, a host handle after [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> HestonHullWhite2F<T, Euler, S> {
  /// Panics unless all are finite, `s0 > 0`, `v0`, `kappa`, `theta`, `sigma`, `sigma1`, `sigma2`
  /// are non-negative, `n >= 2`, a given `t > 0` and `rho` is a `4 × 4` positive-definite correlation.
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
    let chol = Self::check_rho(&rho);
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

/// Holds the short rate's mean at experiment 4.2's 3 percent under `a = 1.1`.
fn flat_drift_target<T: FloatExt>(_t: T) -> T {
  T::from_f64_fast(1.1 * 0.03)
}

/// Grzelak et al. (2011), experiment 4.2 with its matrix (4.4): Feller holds, `2κθ = 0.32 ≥ σ²`;
/// t=1, 252 points.
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

/// The setters chain after [`qe`](Self::qe) too, where they hold QE's preconditions as well.
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

  /// The factor of `rho` in this scheme's driver order; a non-positive pivot rejects the matrix.
  fn check_rho(rho: &Array2<T>) -> Array2<T> {
    let dim = rho.dim();
    assert!(
      dim == (4, 4),
      "rho must satisfy `rho.dim() == (4, 4)`, got rho.dim() = {dim:?}"
    );
    validate_correlation(rho);
    Sch::factor(rho)
  }

  fn check_r0(r0: Option<T>) {
    require!(r0.is_none_or(|r0| r0.is_finite()), r0);
  }

  /// Replace `s0`; panics unless `s0 > 0` and `s0` is finite.
  pub fn with_s0(mut self, s0: T) -> Self {
    Self::check_s0(s0);
    self.s0 = s0;
    self
  }

  /// Replace `v0`; panics unless `v0 >= 0` and `v0` is finite.
  pub fn with_v0(mut self, v0: T) -> Self {
    Self::check_v0(v0);
    self.v0 = v0;
    self
  }

  /// Replace `kappa`; panics unless `kappa >= 0` (`> 0` under QE) and `kappa` is finite.
  pub fn with_kappa(mut self, kappa: T) -> Self {
    Self::check_kappa(kappa);
    self.kappa = kappa;
    Sch::check_parameters(self.kappa, self.theta, self.sigma, &self.rho);
    self
  }

  /// Replace `theta`; panics unless `theta >= 0` (`> 0` under QE) and `theta` is finite.
  pub fn with_theta(mut self, theta: T) -> Self {
    Self::check_theta(theta);
    self.theta = theta;
    Sch::check_parameters(self.kappa, self.theta, self.sigma, &self.rho);
    self
  }

  /// Replace `sigma`; panics unless `sigma >= 0` (`> 0` under QE) and `sigma` is finite.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    Self::check_sigma(sigma);
    self.sigma = sigma;
    Sch::check_parameters(self.kappa, self.theta, self.sigma, &self.rho);
    self
  }

  /// Replace `theta_r`.
  pub fn with_theta_r(mut self, theta_r: impl Into<Fn1D<T>>) -> Self {
    self.theta_r = theta_r.into();
    self
  }

  /// Replace `a`; panics unless `a` is finite.
  pub fn with_a(mut self, a: T) -> Self {
    Self::check_a(a);
    self.a = a;
    self
  }

  /// Replace `sigma1`; panics unless `sigma1 >= 0` and `sigma1` is finite.
  pub fn with_sigma1(mut self, sigma1: T) -> Self {
    Self::check_sigma1(sigma1);
    self.sigma1 = sigma1;
    self
  }

  /// Replace `sigma2`; panics unless `sigma2 >= 0` and `sigma2` is finite.
  pub fn with_sigma2(mut self, sigma2: T) -> Self {
    Self::check_sigma2(sigma2);
    self.sigma2 = sigma2;
    self
  }

  /// Replace `b`; panics unless `b` is finite.
  pub fn with_b(mut self, b: T) -> Self {
    Self::check_b(b);
    self.b = b;
    self
  }

  /// Replace `rho` and its factor; panics unless `rho` is a `4 × 4` positive-definite correlation
  /// matrix, with `rho[(1, 2)]` and `rho[(1, 3)]` zero under QE.
  pub fn with_rho(mut self, rho: Array2<T>) -> Self {
    self.chol = Self::check_rho(&rho);
    self.rho = rho;
    Sch::check_parameters(self.kappa, self.theta, self.sigma, &self.rho);
    self
  }

  /// The correlation matrix of the drivers, in the order `(x, v, r, u)`.
  pub fn rho(&self) -> &Array2<T> {
    &self.rho
  }

  /// Replace `r0`; panics unless `r0`, when given, is finite.
  pub fn with_r0(mut self, r0: Option<T>) -> Self {
    Self::check_r0(r0);
    self.r0 = r0;
    self
  }

  /// Replace `n`; panics unless `n >= 2`.
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self
  }

  /// Replace the horizon `t`; panics unless `t`, when given, is positive and finite.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self
  }

  /// Replace the seed strategy's value.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }

  /// `theta_r` at every grid point, tabulated once: a Python callable costs a GIL round trip.
  fn rate_curve(&self) -> Vec<T> {
    let dt = self.dt();
    (0..self.n)
      .map(|i| self.theta_r.call(T::from_usize_(i) * dt))
      .collect()
  }

  /// Time step `t / (n - 1)`.
  pub fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }
}

impl<T: FloatExt, S: SeedExt, B> HestonHullWhite2F<T, Euler, S, B> {
  /// Re-tag the model with the [`AndersenQe`] variance step; panics unless `kappa`, `theta`,
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
      chol: HestonHullWhite2F::<T, AndersenQe, S, B>::check_rho(&self.rho),
      rho: self.rho,
      r0: self.r0,
      n: self.n,
      t: self.t,
      seed: self.seed,
      _scheme: PhantomData,
      backend: self.backend,
    };
    AndersenQe::check_parameters(model.kappa, model.theta, model.sigma, &model.rho);
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
      curve: self.rate_curve(),
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
  curve: Vec<T>,
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
    Sch::simulate_into(self.model, &self.curve, &self.seed, out);
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
