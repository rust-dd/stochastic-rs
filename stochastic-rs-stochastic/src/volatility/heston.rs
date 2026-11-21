//! Heston stochastic volatility, by the `Euler` or `AndersenQe` scheme chosen at compile time.
//!
//! $$
//! \begin{aligned}dS_t&=\mu S_tdt+\sqrt{v_t}S_tdW_t^S\\
//! dv_t&=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dW_t^v,\ d\langle W^S,W^v\rangle_t=\rho dt\end{aligned}
//! $$
//!
use std::marker::PhantomData;

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::HestonPow;
use crate::device::Cpu;
use crate::noise::cgns::Cgns;
use crate::params::VarianceBoundary;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

mod scheme;
pub use scheme::AndersenQe;
pub use scheme::Euler;
pub use scheme::HestonScheme;

#[derive(Debug, Clone)]
pub struct Heston<T: FloatExt, Sch: HestonScheme = Euler, S: SeedExt = Unseeded, B = Cpu> {
  /// Initial spot.
  pub s0: T,
  /// Initial variance, not a volatility: `sqrt(v_t)` is what `dS_t` reads.
  pub v0: T,
  /// Mean reversion rate
  pub kappa: T,
  /// Long-run variance level (θ in the module header's `dv_t` equation) —
  /// a variance, not a volatility, for the same reason as `v0`.
  pub theta: T,
  /// Volatility of volatility
  pub sigma: T,
  /// Correlation between the stock price and its volatility
  pub rho: T,
  /// Drift of the stock price
  pub mu: T,
  /// Grid points sampled along the path; `dt = t / (n − 1)`.
  pub n: usize,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Power of the variance: 0.5 is the original Heston model, 1.5 the 3/2 model.
  pub pow: HestonPow,
  /// How a negative square-root state is treated after a step; Feller's condition is not required.
  pub boundary: VarianceBoundary,
  /// [`Unseeded`] or a [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// Noise generator (used by the [`Euler`] scheme; [`AndersenQe`] draws its
  /// own noise and leaves this untouched).
  cgns: Cgns<T>,
  /// Zero-sized marker for the compile-time variance scheme.
  _scheme: PhantomData<Sch>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Heston<T, Euler, S> {
  /// Panics unless `v0 >= 0`, `kappa >= 0`, `theta >= 0`, `sigma >= 0`, `-1 <= rho <= 1`, `n >= 2`,
  /// a given `t` is positive and finite, and every parameter is finite.
  pub fn new(
    s0: T,
    v0: T,
    kappa: T,
    theta: T,
    sigma: T,
    rho: T,
    mu: T,
    n: usize,
    t: Option<T>,
    pow: HestonPow,
    boundary: VarianceBoundary,
    seed: S,
  ) -> Self {
    Self::check_s0(s0);
    Self::check_v0(v0);
    Self::check_kappa(kappa);
    Self::check_theta(theta);
    Self::check_sigma(sigma);
    Self::check_rho(rho);
    Self::check_mu(mu);
    check_n(n);
    check_t(t);

    Self {
      backend: Cpu,
      s0,
      v0,
      kappa,
      theta,
      sigma,
      rho,
      mu,
      n,
      t,
      pow,
      boundary,
      seed,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
      _scheme: PhantomData,
    }
  }
}

/// s₀=100, v₀=0.04, κ=2.0, θ=0.04, σ=0.3, ρ=-0.7, μ=0.05, a Heston (1993) model inside Feller
/// (`2κθ = 0.16 ≥ σ² = 0.09`); t=1, 252 points, a trading year of daily closes.
impl<T: FloatExt> Default for Heston<T, Euler, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(100.0),
      T::from_f64_fast(0.04),
      T::from_f64_fast(2.0),
      T::from_f64_fast(0.04),
      T::from_f64_fast(0.3),
      T::from_f64_fast(-0.7),
      T::from_f64_fast(0.05),
      252,
      Some(T::one()),
      HestonPow::Sqrt,
      VarianceBoundary::default(),
      Unseeded,
    )
  }
}

/// Generic over `Sch: HestonScheme`, since `cgns` and every other field are scheme-independent:
/// the setters chain after [`qe()`](Heston::qe) too.
impl<T: FloatExt, Sch: HestonScheme, S: SeedExt, B> Heston<T, Sch, S, B> {
  fn check_s0(s0: T) {
    require!(s0.is_finite(), s0);
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

  fn check_rho(rho: T) {
    require!(rho.is_finite(), rho);
  }

  fn check_mu(mu: T) {
    require!(mu.is_finite(), mu);
  }

  /// Replace `s0`; panics unless `s0` is finite.
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

  /// Replace `kappa`; panics unless `kappa >= 0` and `kappa` is finite.
  pub fn with_kappa(mut self, kappa: T) -> Self {
    Self::check_kappa(kappa);
    self.kappa = kappa;
    self
  }

  /// Replace `theta`; panics unless `theta >= 0` and `theta` is finite.
  pub fn with_theta(mut self, theta: T) -> Self {
    Self::check_theta(theta);
    self.theta = theta;
    self
  }

  /// Replace `sigma`; panics unless `sigma >= 0` and `sigma` is finite.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    Self::check_sigma(sigma);
    self.sigma = sigma;
    self
  }

  /// Replace `rho` and rebuild the cached correlated-Gaussian generator `cgns`, which the sampler
  /// reads; panics unless `-1 <= rho <= 1`.
  pub fn with_rho(mut self, rho: T) -> Self {
    Self::check_rho(rho);
    self.rho = rho;
    self.cgns = Cgns::new(rho, self.n - 1, self.t, Unseeded);
    self
  }

  /// Replace `mu`; panics unless `mu` is finite.
  pub fn with_mu(mut self, mu: T) -> Self {
    Self::check_mu(mu);
    self.mu = mu;
    self
  }

  /// Replace `n`, the grid points along the path; rebuilds the cached correlated-Gaussian
  /// generator, whose length and step size derive from `n`; panics unless `n >= 2`.
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self.cgns = Cgns::new(self.rho, n - 1, self.t, Unseeded);
    self
  }

  /// Replace the simulation horizon `t`; rebuilds the cached correlated-Gaussian generator's step
  /// size, which derives from `t`; panics unless `t`, when given, is positive and finite.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self.cgns = Cgns::new(self.rho, self.n - 1, t, Unseeded);
    self
  }

  /// Replace `pow`.
  pub fn with_pow(mut self, pow: HestonPow) -> Self {
    self.pow = pow;
    self
  }

  /// Replace `boundary`.
  pub fn with_boundary(mut self, boundary: VarianceBoundary) -> Self {
    self.boundary = boundary;
    self
  }

  /// Replace the seed strategy's value.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }

  /// QE's preconditions, checked at the switch and again where the scheme runs, since the fields
  /// stay public.
  fn assert_qe_parameters(&self) {
    assert!(
      matches!(self.pow, HestonPow::Sqrt),
      "pow must satisfy `matches!(pow, HestonPow::Sqrt)`, got pow = {:?}",
      self.pow
    );
    assert!(
      self.kappa > T::zero(),
      "kappa must satisfy `kappa > T::zero()`, got kappa = {:?}",
      self.kappa
    );
    assert!(
      self.sigma > T::zero(),
      "sigma must satisfy `sigma > T::zero()`, got sigma = {:?}",
      self.sigma
    );
    assert!(
      self.s0 > T::zero(),
      "s0 must satisfy `s0 > T::zero()`, got s0 = {:?}",
      self.s0
    );
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> Heston<T, Euler, S, B> {
  /// Re-tag the model with the [`AndersenQe`] variance scheme at no runtime cost; panics unless
  /// `pow` is `HestonPow::Sqrt` and `kappa`, `sigma` and `s0` are positive.
  pub fn qe(self) -> Heston<T, AndersenQe, S, B> {
    self.assert_qe_parameters();
    Heston {
      backend: self.backend,
      s0: self.s0,
      v0: self.v0,
      kappa: self.kappa,
      theta: self.theta,
      sigma: self.sigma,
      rho: self.rho,
      mu: self.mu,
      n: self.n,
      t: self.t,
      pow: self.pow,
      boundary: self.boundary,
      seed: self.seed,
      cgns: self.cgns,
      _scheme: PhantomData,
    }
  }
}

/// The Euler engine's view of the Euler scheme: the variance's exponent travels as a parameter,
/// while the [`VarianceBoundary`] picks the family, as for every square-root diffusion here.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for Heston<T, Euler, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let pow_v = match self.pow {
      HestonPow::Sqrt => T::from_f64_fast(0.5),
      HestonPow::ThreeHalves => T::from_f64_fast(1.5),
    };
    let (mu, kappa, theta, sigma, rho) = (self.mu, self.kappa, self.theta, self.sigma, self.rho);
    match self.boundary {
      VarianceBoundary::FullTruncation => {
        crate::euler::EulerSpec::HestonFullTruncation([mu, kappa, theta, sigma, rho, pow_v])
      }
      VarianceBoundary::Absorption => crate::euler::EulerSpec::Heston {
        mu: self.mu,
        kappa: self.kappa,
        theta: self.theta,
        sigma: self.sigma,
        rho: self.rho,
        pow_v,
      },
      VarianceBoundary::Reflect => crate::euler::EulerSpec::HestonReflected {
        mu: self.mu,
        kappa: self.kappa,
        theta: self.theta,
        sigma: self.sigma,
        rho: self.rho,
        pow_v,
      },
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [self.s0, self.v0.max(T::zero()), T::zero(), T::zero()]
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  /// The correlated-noise source divides the horizon by the number of points,
  /// not by the number of steps, so the device steps by that same amount.
  fn time_step(&self) -> T {
    self.cgns.dt()
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, Sch: HestonScheme, S: SeedExt] Heston<T, Sch, S> { s0, v0, kappa, theta, sigma, rho, mu, n, t, pow, boundary, seed, cgns, _scheme } via euler);

/// Derives a seed once, at construction, for [`HestonSampler`] to own and pass into
/// `Sch::simulate`: the mixed next tick hash-scrambles chunk `i`'s basis against chunk `i+1`'s.
macro_rules! heston_sampler_impl {
  ($scheme:ty) => {
    fn sampler(&self) -> HestonSampler<'_, T, $scheme, S, B> {
      HestonSampler {
        model: self,
        seed: self.seed.derive(),
      }
    }
  };
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Heston<T, Euler, S, B>
{
}

/// The Euler scheme is the one the engine reproduces, so this half of the
/// process routes through the backend.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for Heston<T, Euler, S, B>
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = HestonSampler<'s, T, Euler, S, B>
  where
    Self: 's;

  heston_sampler_impl!(Euler);

  /// Through the Euler engine: a device steps both components in its kernel, the host runs this
  /// process's own scheme, chunked as `ProcessExt` chunks.
  fn sample(&self) -> [Array1<T>; 2] {
    self.backend.system_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&[Array1<T>; 2]) -> R + Sync) -> Vec<R> {
    self.backend.system_paths_map(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<[Array1<T>; 2]> {
    self.backend.system_paths(self, m)
  }

  fn try_sample(&self) -> Result<[Array1<T>; 2], crate::device::DeviceError> {
    self.backend.try_system_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<[Array1<T>; 2]>, crate::device::DeviceError> {
    self.backend.try_system_paths(self, m)
  }
}

/// The Euler engine's view of the quadratic-exponential scheme: the variance is drawn from a
/// moment-matched law whose branch needs a uniform, supplied per step; the constants fold here.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for Heston<T, AndersenQe, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    self.assert_qe_parameters();
    let dt = self.time_step();
    let (one, two, half) = (T::one(), T::from_f64_fast(2.0), T::from_f64_fast(0.5));
    let e_kd = (-self.kappa * dt).exp();
    let eps = self.sigma;
    let krho_eps = self.kappa * self.rho / eps;
    crate::euler::EulerSpec::AndersenQe {
      theta: self.theta,
      e_kd,
      c1: eps * eps * e_kd / self.kappa,
      c2: self.theta * eps * eps / (two * self.kappa),
      k0: -self.rho * self.kappa * self.theta * dt / eps,
      k1: half * dt * (krho_eps - half) - self.rho / eps,
      k2: half * dt * (krho_eps - half) + self.rho / eps,
      k34: half * dt * (one - self.rho * self.rho),
      mu: self.mu,
    }
  }

  /// The scheme evolves the log spot, so the first slot is `ln S₀` and the
  /// report exponentiates it back.
  fn initial_state(&self) -> [T; 4] {
    self.assert_qe_parameters();
    [self.s0.ln(), self.v0.max(T::zero()), T::zero(), T::zero()]
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Heston<T, AndersenQe, S, B>
{
}

/// The quadratic-exponential scheme reaches a device through its own family,
/// so this half routes through the backend as the Euler half does.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for Heston<T, AndersenQe, S, B>
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = HestonSampler<'s, T, AndersenQe, S, B>
  where
    Self: 's;

  heston_sampler_impl!(AndersenQe);

  fn sample(&self) -> [Array1<T>; 2] {
    self.backend.system_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&[Array1<T>; 2]) -> R + Sync) -> Vec<R> {
    self.backend.system_paths_map(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<[Array1<T>; 2]> {
    self.backend.system_paths(self, m)
  }

  fn try_sample(&self) -> Result<[Array1<T>; 2], crate::device::DeviceError> {
    self.backend.try_system_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<[Array1<T>; 2]>, crate::device::DeviceError> {
    self.backend.try_system_paths(self, m)
  }
}

/// Reusable [`Heston`] sampler: borrows the process and owns a seed derived once. Each call
/// dispatches to `Sch::simulate`, which sets up its own RNG from that seed.
#[doc(hidden)]
#[derive(Debug)]
pub struct HestonSampler<'a, T: FloatExt, Sch: HestonScheme, S: SeedExt, B> {
  model: &'a Heston<T, Sch, S, B>,
  seed: S,
}

impl<T: FloatExt, Sch: HestonScheme, S: SeedExt, B: Send + Sync> crate::traits::Sealed
  for HestonSampler<'_, T, Sch, S, B>
{
}

impl<T: FloatExt, Sch: HestonScheme, S: SeedExt, B: Send + Sync> PathSampler<T>
  for HestonSampler<'_, T, Sch, S, B>
{
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    *out = Sch::simulate(self.model, &self.seed);
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    Sch::simulate(self.model, &self.seed)
  }
}

mod malliavin;

#[cfg(test)]
mod tests;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyHeston;
