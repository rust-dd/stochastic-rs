//! # Heston
//!
//! $$
//! \begin{aligned}dS_t&=\mu S_tdt+\sqrt{v_t}S_tdW_t^S\\dv_t&=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dW_t^v,\ d\langle W^S,W^v\rangle_t=\rho dt\end{aligned}
//! $$
//!
use std::marker::PhantomData;

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::HestonPow;
use crate::device::Cpu;
use crate::malliavin::Malliavin;
use crate::noise::cgns::Cgns;
use crate::params::VarianceBoundary;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

mod scheme;
pub use scheme::AndersenQe;
pub use scheme::Euler;
pub use scheme::HestonScheme;

#[derive(Clone)]
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
  /// Number of time steps
  pub n: usize,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Power of the variance
  /// If 0.5 then it is the original Heston model
  /// If 1.5 then it is the 3/2 model
  pub pow: HestonPow,
  /// Treatment of a negative square-root state after a step; the Feller condition is not required.
  pub boundary: VarianceBoundary,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
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
    assert!(
      s0.is_finite(),
      "s0 must satisfy `s0.is_finite()`, got s0 = {s0:?}"
    );
    assert!(
      v0.is_finite(),
      "v0 must satisfy `v0.is_finite()`, got v0 = {v0:?}"
    );
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
    assert!(
      rho.is_finite(),
      "rho must satisfy `rho.is_finite()`, got rho = {rho:?}"
    );
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(
      kappa >= T::zero(),
      "kappa must satisfy `kappa >= T::zero()`, got kappa = {kappa:?}"
    );
    assert!(
      theta >= T::zero(),
      "theta must satisfy `theta >= T::zero()`, got theta = {theta:?}"
    );
    assert!(
      sigma >= T::zero(),
      "sigma must satisfy `sigma >= T::zero()`, got sigma = {sigma:?}"
    );
    assert!(
      v0 >= T::zero(),
      "v0 must satisfy `v0 >= T::zero()`, got v0 = {v0:?}"
    );

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

/// s₀=100, v₀=0.04, κ=2.0, θ=0.04, σ=0.3, ρ=-0.7, μ=0.05 — a
/// parameterization of the Heston (1993) stochastic-volatility model
/// satisfying the Feller condition (`2κθ = 0.16 ≥ σ² = 0.09`). t=1, n=252
/// — one trading year of daily steps (this crate's `Default` convention).
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
      VarianceBoundary::Floor,
      Unseeded,
    )
  }
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Heston::default().with_kappa(3.0).with_rho(-0.8)`. Implemented
/// generically over `Sch: HestonScheme` (not just [`Euler`], the only
/// scheme `new()`/`Default` produce) since `cgns` and every other field are
/// scheme-independent — this is what lets the setters chain after
/// [`qe()`](Heston::qe) too, e.g.
/// `Heston::default().qe().with_kappa(3.0)`.
impl<T: FloatExt, Sch: HestonScheme, S: SeedExt, B> Heston<T, Sch, S, B> {
  /// Replace `s0`, all else unchanged.
  pub fn with_s0(mut self, s0: T) -> Self {
    self.s0 = s0;
    self
  }

  /// Replace `v0`, all else unchanged.
  pub fn with_v0(mut self, v0: T) -> Self {
    assert!(
      v0 >= T::zero(),
      "v0 must satisfy `v0 >= T::zero()`, got v0 = {v0:?}"
    );
    self.v0 = v0;
    self
  }

  /// Replace `kappa`, all else unchanged.
  pub fn with_kappa(mut self, kappa: T) -> Self {
    assert!(
      kappa >= T::zero(),
      "kappa must satisfy `kappa >= T::zero()`, got kappa = {kappa:?}"
    );
    self.kappa = kappa;
    self
  }

  /// Replace `theta`, all else unchanged.
  pub fn with_theta(mut self, theta: T) -> Self {
    assert!(
      theta >= T::zero(),
      "theta must satisfy `theta >= T::zero()`, got theta = {theta:?}"
    );
    self.theta = theta;
    self
  }

  /// Replace `sigma`, all else unchanged.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    assert!(
      sigma >= T::zero(),
      "sigma must satisfy `sigma >= T::zero()`, got sigma = {sigma:?}"
    );
    self.sigma = sigma;
    self
  }

  /// Replace `rho`; rebuilds the cached correlated-Gaussian generator
  /// (`cgns`) so the new correlation actually reaches the sampler instead
  /// of a stale one computed from the old `rho`.
  pub fn with_rho(mut self, rho: T) -> Self {
    self.rho = rho;
    self.cgns = Cgns::new(rho, self.n - 1, self.t, Unseeded);
    self
  }

  /// Replace `mu`, all else unchanged.
  pub fn with_mu(mut self, mu: T) -> Self {
    self.mu = mu;
    self
  }

  /// Replace the number of simulation steps `n`; rebuilds the cached
  /// correlated-Gaussian generator, whose length and step size derive
  /// from `n`.
  pub fn with_steps(mut self, n: usize) -> Self {
    self.n = n;
    self.cgns = Cgns::new(self.rho, n - 1, self.t, Unseeded);
    self
  }

  /// Replace the simulation horizon `t`; rebuilds the cached
  /// correlated-Gaussian generator's step size, which derives from `t`.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    self.cgns = Cgns::new(self.rho, self.n - 1, t, Unseeded);
    self
  }

  /// Replace `pow`, all else unchanged.
  pub fn with_pow(mut self, pow: HestonPow) -> Self {
    self.pow = pow;
    self
  }

  /// Replace `boundary`, all else unchanged.
  pub fn with_boundary(mut self, boundary: VarianceBoundary) -> Self {
    self.boundary = boundary;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }

  /// QE's preconditions, checked at the switch and again where the scheme runs, since the fields stay public.
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
    match self.boundary {
      VarianceBoundary::Floor => crate::euler::EulerSpec::Heston {
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

/// Derives a seed once, at construction, for [`HestonSampler`] to own and
/// pass into `Sch::simulate` — see that trait method's docs for why.
/// Deriving (not cloning) is what decorrelates chunks: the derived value
/// is `self.seed`'s *mixed* next tick, not a raw snapshot, so chunk `i`'s
/// basis and chunk `i+1`'s basis are hash-scrambled relative to each
/// other rather than one raw stride apart.
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

  /// Through the Euler engine: on a device both components step in the
  /// kernel, on the host devices it is this process's own scheme, chunked
  /// exactly as `ProcessExt` chunks.
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

/// The Euler engine's view of the quadratic-exponential scheme. The variance
/// is drawn from a moment-matched law rather than stepped, and the branch it
/// picks needs a uniform, which the kernel supplies per step; every constant
/// the scheme hoists out of its own path loop is folded here for the same
/// reason it hoists them.
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

/// Reusable [`Heston`] sampler: borrows the process and owns a seed derived
/// once at construction. The variance discretisation runs inside the
/// compile-time-selected [`HestonScheme`], which owns its own RNG setup
/// beyond that seed, so each call re-dispatches to `Sch::simulate`; there is
/// nothing else reusable to hoist across calls.
#[doc(hidden)]
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

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> Heston<T, Euler, S, B> {
  /// The path and a frozen-variance closed form of the variance's derivative at each `t_i`:
  /// `σ v_T^p / 2 · exp(c (n − i) Δt)`, `c` Alòs and Ewald's (2008) rate at `v_T` for `p = 1/2`.
  pub fn malliavin(&self) -> Malliavin<T, [Array1<T>; 2]> {
    let [s, v] = self.sample();
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1);

    let mut det_term = Array1::zeros(self.n);
    let mut malliavin = Array1::zeros(self.n);
    let f2 = T::from_usize_(2);

    for i in 0..self.n {
      match self.pow {
        HestonPow::Sqrt => {
          det_term[i] = ((-(self.kappa * self.theta / f2
            - self.sigma.powi(2) / T::from_usize_(8))
            * (T::one() / *v.last().unwrap())
            - self.kappa / f2)
            * (T::from_usize_(self.n - i) * dt))
            .exp();
          malliavin[i] = (self.sigma * v.last().unwrap().sqrt() / f2) * det_term[i];
        }
        HestonPow::ThreeHalves => {
          det_term[i] = ((-(self.kappa * self.theta / f2
            + T::from_usize_(3) * self.sigma.powi(2) / T::from_usize_(8))
            * *v.last().unwrap()
            - (self.kappa * self.theta) / f2)
            * (T::from_usize_(self.n - i) * dt))
            .exp();
          malliavin[i] =
            (self.sigma * v.last().unwrap().powf(T::from_f64_fast(1.5)) / f2) * det_term[i];
        }
      };
    }

    Malliavin {
      paths: [s, v],
      derivative: malliavin,
    }
  }
}

#[cfg(test)]
mod tests;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyHeston;
