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
//! approximation the paper prices with. The output is the `5 × n` matrix `[S, v, r, u, ∫r]`.
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

use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::linalg::validate_correlation;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

mod scheme;
pub use scheme::HybridScheme;

pub use crate::volatility::heston::AndersenQe;
pub use crate::volatility::heston::Euler;

/// Rows of the output: spot, variance, short rate, auxiliary rate factor, integrated short rate.
pub const ROWS: usize = 5;

#[derive(Clone)]
pub struct HestonHullWhite2F<T: FloatExt, S: SeedExt = Unseeded, Sch: HybridScheme = Euler, B = Cpu>
{
  /// Mean-reversion speed κ of the variance.
  pub kappa: T,
  /// Long-run variance θ, a variance and not a volatility; also the default start `v0`.
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
  /// Grid points including `t = 0`, at least 2.
  pub n: usize,
  /// Initial spot (defaults to 1).
  pub s0: Option<T>,
  /// Initial variance (defaults to `theta`).
  pub v0: Option<T>,
  /// Initial short rate (defaults to 0); `u` starts at 0.
  pub r0: Option<T>,
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

/// `name` must be finite and non-negative; the hybrid's rates and volatilities are checked this way.
fn assert_non_negative<T: FloatExt>(name: &str, x: T) {
  assert!(
    x.is_finite() && x >= T::zero(),
    "{name} must satisfy `{name}.is_finite() && {name} >= T::zero()`, got {name} = {x:?}"
  );
}

/// `rho` must be a `4 × 4` correlation matrix; the Cholesky factorisation rejects an indefinite one.
fn checked_factor<T: FloatExt, Sch: HybridScheme>(rho: &Array2<T>) -> Array2<T> {
  assert_eq!(
    rho.dim(),
    (4, 4),
    "rho must satisfy `rho.dim() == (4, 4)`, got rho.dim() = {:?}",
    rho.dim()
  );
  validate_correlation(rho);
  Sch::factor(rho)
}

fn assert_finite<T: FloatExt>(name: &str, x: T) {
  assert!(
    x.is_finite(),
    "{name} must satisfy `{name}.is_finite()`, got {name} = {x:?}"
  );
}

fn assert_positive<T: FloatExt>(name: &str, x: T) {
  assert!(
    x.is_finite() && x > T::zero(),
    "{name} must satisfy `{name}.is_finite() && {name} > T::zero()`, got {name} = {x:?}"
  );
}

impl<T: FloatExt, S: SeedExt> HestonHullWhite2F<T, S> {
  /// Panics if a parameter is not finite, `kappa`, `theta`, `sigma`, `sigma1`, `sigma2` or `v0` is
  /// negative, `s0` or `t` is not positive, `n < 2`, or `rho` is not a `4 × 4` positive-definite correlation matrix.
  pub fn new(
    kappa: T,
    theta: T,
    sigma: T,
    theta_r: impl Into<Fn1D<T>>,
    a: T,
    sigma1: T,
    sigma2: T,
    b: T,
    rho: Array2<T>,
    n: usize,
    s0: Option<T>,
    v0: Option<T>,
    r0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert_non_negative("kappa", kappa);
    assert_non_negative("theta", theta);
    assert_non_negative("sigma", sigma);
    assert_finite("a", a);
    assert_non_negative("sigma1", sigma1);
    assert_non_negative("sigma2", sigma2);
    assert_finite("b", b);
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n}");
    if let Some(s0) = s0 {
      assert_positive("s0", s0);
    }
    if let Some(v0) = v0 {
      assert_non_negative("v0", v0);
    }
    if let Some(r0) = r0 {
      assert_finite("r0", r0);
    }
    if let Some(t) = t {
      assert_positive("t", t);
    }
    let chol = checked_factor::<T, Euler>(&rho);
    Self {
      kappa,
      theta,
      sigma,
      theta_r: theta_r.into(),
      a,
      sigma1,
      sigma2,
      b,
      rho,
      n,
      s0,
      v0,
      r0,
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
impl<T: FloatExt> Default for HestonHullWhite2F<T, Unseeded, Euler> {
  fn default() -> Self {
    let f = T::from_f64_fast;
    let rho = ndarray::array![
      [f(1.0), f(-0.3), f(0.35), f(0.08)],
      [f(-0.3), f(1.0), f(0.0), f(0.0)],
      [f(0.35), f(0.0), f(1.0), f(-0.4)],
      [f(0.08), f(0.0), f(-0.4), f(1.0)],
    ];
    Self::new(
      f(0.8),
      f(0.2),
      f(0.2),
      flat_drift_target::<T> as fn(T) -> T,
      f(1.1),
      f(0.01),
      f(0.015),
      f(0.8),
      rho,
      252,
      Some(T::one()),
      Some(f(0.2)),
      Some(f(0.03)),
      Some(T::one()),
      Unseeded,
    )
  }
}

/// Every field has a matching `with_*` setter, checked as `new` checks it, chaining after [`qe`](Self::qe) too.
impl<T: FloatExt, S: SeedExt, Sch: HybridScheme, B> HestonHullWhite2F<T, S, Sch, B> {
  pub fn with_kappa(mut self, kappa: T) -> Self {
    assert_non_negative("kappa", kappa);
    self.kappa = kappa;
    self
  }

  pub fn with_theta(mut self, theta: T) -> Self {
    assert_non_negative("theta", theta);
    self.theta = theta;
    self
  }

  pub fn with_sigma(mut self, sigma: T) -> Self {
    assert_non_negative("sigma", sigma);
    self.sigma = sigma;
    self
  }

  pub fn with_theta_r(mut self, theta_r: impl Into<Fn1D<T>>) -> Self {
    self.theta_r = theta_r.into();
    self
  }

  pub fn with_a(mut self, a: T) -> Self {
    assert_finite("a", a);
    self.a = a;
    self
  }

  pub fn with_sigma1(mut self, sigma1: T) -> Self {
    assert_non_negative("sigma1", sigma1);
    self.sigma1 = sigma1;
    self
  }

  pub fn with_sigma2(mut self, sigma2: T) -> Self {
    assert_non_negative("sigma2", sigma2);
    self.sigma2 = sigma2;
    self
  }

  pub fn with_b(mut self, b: T) -> Self {
    assert_finite("b", b);
    self.b = b;
    self
  }

  /// Replaces the correlation matrix and refactors it in this scheme's driver order.
  pub fn with_rho(mut self, rho: Array2<T>) -> Self {
    self.chol = checked_factor::<T, Sch>(&rho);
    self.rho = rho;
    self
  }

  pub fn with_steps(mut self, n: usize) -> Self {
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n}");
    self.n = n;
    self
  }

  pub fn with_s0(mut self, s0: Option<T>) -> Self {
    if let Some(s0) = s0 {
      assert_positive("s0", s0);
    }
    self.s0 = s0;
    self
  }

  pub fn with_v0(mut self, v0: Option<T>) -> Self {
    if let Some(v0) = v0 {
      assert_non_negative("v0", v0);
    }
    self.v0 = v0;
    self
  }

  pub fn with_r0(mut self, r0: Option<T>) -> Self {
    if let Some(r0) = r0 {
      assert_finite("r0", r0);
    }
    self.r0 = r0;
    self
  }

  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    if let Some(t) = t {
      assert_positive("t", t);
    }
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
    assert_positive("kappa", self.kappa);
    assert_positive("theta", self.theta);
    assert_positive("sigma", self.sigma);
    for (what, index) in [("rho[(1, 2)]", (1, 2)), ("rho[(1, 3)]", (1, 3))] {
      let value = self.rho[index];
      assert!(
        value == T::zero(),
        "the rate-variance correlation {what} must satisfy `{what} == T::zero()` under QE, got {what} = {value:?}"
      );
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> HestonHullWhite2F<T, S, Euler, B> {
  /// Re-tags the model with the [`AndersenQe`] variance step; panics unless `kappa`, `theta` and
  /// `sigma` are positive and the rate-variance correlations `rho[(1, 2)]`, `rho[(1, 3)]` are zero.
  pub fn qe(self) -> HestonHullWhite2F<T, S, AndersenQe, B> {
    let model = HestonHullWhite2F {
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
      n: self.n,
      s0: self.s0,
      v0: self.v0,
      r0: self.r0,
      t: self.t,
      seed: self.seed,
      _scheme: PhantomData,
      backend: self.backend,
    };
    model.assert_qe_parameters();
    model
  }
}

backend_switch!([T: FloatExt, S: SeedExt, Sch: HybridScheme] HestonHullWhite2F<T, S, Sch> { kappa, theta, sigma, theta_r, a, sigma1, sigma2, b, rho, n, s0, v0, r0, t, seed, chol, _scheme } via host);

impl<T: FloatExt, S: SeedExt, Sch: HybridScheme, B: crate::device::HostBackend>
  crate::traits::Sealed for HestonHullWhite2F<T, S, Sch, B>
{
}

impl<T: FloatExt, S: SeedExt, Sch: HybridScheme, B: crate::device::HostBackend> ProcessExt<T>
  for HestonHullWhite2F<T, S, Sch, B>
{
  type Output = Array2<T>;
  type Sampler<'s>
    = HestonHullWhite2FSampler<'s, T, S, Sch, B>
  where
    Self: 's;

  /// Derives (not clones) `self.seed`, so every chunk of `sample_par` owns a hash-mixed basis.
  fn sampler(&self) -> HestonHullWhite2FSampler<'_, T, S, Sch, B> {
    HestonHullWhite2FSampler {
      model: self,
      seed: self.seed.derive(),
    }
  }

  /// Five state rows and four shocks exceed the engine's four slots, so every configuration samples here.
  fn device_fallback(&self) -> Option<&'static str> {
    Some("five state rows, one more than the kernels' four slots")
  }
}

/// Reusable [`HestonHullWhite2F`] sampler: borrows the model for `theta_r` and owns a seed derived
/// once, which the scheme's streams advance on every path of the chunk.
#[doc(hidden)]
pub struct HestonHullWhite2FSampler<'a, T: FloatExt, S: SeedExt, Sch: HybridScheme, B> {
  model: &'a HestonHullWhite2F<T, S, Sch, B>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt, Sch: HybridScheme, B: Send + Sync> crate::traits::Sealed
  for HestonHullWhite2FSampler<'_, T, S, Sch, B>
{
}

impl<T: FloatExt, S: SeedExt, Sch: HybridScheme, B: Send + Sync> PathSampler<T>
  for HestonHullWhite2FSampler<'_, T, S, Sch, B>
{
  type Output = Array2<T>;

  fn sample_into(&mut self, out: &mut Array2<T>) {
    Sch::simulate_into(self.model, &self.seed, out);
  }

  fn sample(&mut self) -> Array2<T> {
    let mut out = Array2::<T>::zeros((ROWS, self.model.n));
    self.sample_into(&mut out);
    out
  }
}

#[cfg(test)]
mod tests;
