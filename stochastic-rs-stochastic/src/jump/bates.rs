//! # Bates
//!
//! $$
//! \begin{aligned}dS_t&=(r-r_f-\lambda k)S_tdt+\sqrt{v_t}S_t dW_t^S+(Y-1)S_{t^-}dN_t\\dv_t&=\kappa(\theta-v_t)dt+\sigma\sqrt{v_t}dW_t^v\end{aligned}
//! $$
//!
//! Reference: Bates D. S. (1996) — *Jumps and Stochastic Volatility:
//! Exchange Rate Processes Implicit in Deutsche Mark Options*, Review of
//! Financial Studies 9(1), 69–107, DOI: 10.1093/rfs/9.1.69.
//!

use std::any::Any;

use ndarray::Array1;
use rand::distr::Distribution;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::device::Cpu;
use crate::noise::cgns::Cgns;
use crate::params::Drift;
use crate::params::VarianceBoundary;
use crate::process::cpoisson::JumpDriver;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// The Bates (1996) model for any law `D` of the relative jump `Y − 1`, `SimdNormal` by default.
/// The paper's [`LogNormalJump`], exact in log space, steps `ln S` so `S > 0`; others step `S`.
#[derive(Clone)]
pub struct Bates<T, D = SimdNormal<T>, S: SeedExt = Unseeded, B = Cpu>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// The asset's drift before the `λk` compensator: `μ`, a cost of carry, or `r − r_f`.
  pub drift: Drift<T>,
  /// Jump (Poisson) intensity λ of the relative jumps, the one rate the arrivals and the drift's
  /// `−λk` compensator read.
  pub lambda: T,
  /// The compensator `E[Y − 1]` of the jump law, subtracted from the drift times `lambda`; only
  /// that value keeps the price a martingale ([`LogNormalJump::mean`] for the lognormal law).
  pub k: T,
  /// Variance-drift intercept (κθ combined) in the reparametrized variance
  /// recursion `dv = (alpha − beta·v)dt + ...`, equivalent to `κ(θ−v)`
  /// with `alpha = κθ`.
  pub alpha: T,
  /// Variance-drift slope (mean-reversion speed κ) in the same
  /// reparametrized recursion.
  pub beta: T,
  /// Vol-of-vol σ scaling the variance factor's own diffusion.
  pub sigma: T,
  /// Instantaneous correlation ρ between the asset's and variance's
  /// driving Brownian motions.
  pub rho: T,
  /// Number of points sampled along the Bates path.
  pub n: usize,
  /// Initial spot.
  pub s0: T,
  /// Initial variance.
  pub v0: T,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Treatment of a negative square-root state after a step; the Feller condition is not required.
  pub boundary: VarianceBoundary,
  /// Correlated-Gaussian generator of the price/variance diffusion; the sampler drives it through
  /// `cgns.sample_impl(&self.seed)`, so its own `Unseeded` seed is never read.
  cgns: Cgns<T>,
  jumps: JumpDriver<D, S>,
  /// The diffusion's seed; the jumps draw from a child of it taken at construction.
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T, D, S: SeedExt> Bates<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Jumps of law `jump_dist` arriving at rate `lambda`, their seed a child of `seed`.
  pub fn new(
    drift: Drift<T>,
    lambda: T,
    k: T,
    alpha: T,
    beta: T,
    sigma: T,
    rho: T,
    jump_dist: D,
    n: usize,
    s0: T,
    v0: T,
    t: Option<T>,
    boundary: VarianceBoundary,
    seed: S,
  ) -> Self {
    drift.assert_finite();
    assert!(
      lambda.is_finite(),
      "lambda must satisfy `lambda.is_finite()`, got lambda = {lambda:?}"
    );
    assert!(
      lambda >= T::zero(),
      "lambda must satisfy `lambda >= T::zero()`, got lambda = {lambda:?}"
    );
    assert!(
      k.is_finite(),
      "k must satisfy `k.is_finite()`, got k = {k:?}"
    );
    assert!(
      alpha.is_finite(),
      "alpha must satisfy `alpha.is_finite()`, got alpha = {alpha:?}"
    );
    assert!(
      beta.is_finite(),
      "beta must satisfy `beta.is_finite()`, got beta = {beta:?}"
    );
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    assert!(
      rho.is_finite(),
      "rho must satisfy `rho.is_finite()`, got rho = {rho:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      s0.is_finite(),
      "s0 must satisfy `s0.is_finite()`, got s0 = {s0:?}"
    );
    assert!(
      v0.is_finite(),
      "v0 must satisfy `v0.is_finite()`, got v0 = {v0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(
      v0 >= T::zero(),
      "v0 must satisfy `v0 >= T::zero()`, got v0 = {v0:?}"
    );

    Self {
      backend: Cpu,
      drift,
      lambda,
      k,
      alpha,
      beta,
      sigma,
      rho,
      n,
      s0,
      v0,
      t,
      boundary,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
      jumps: JumpDriver::new(jump_dist, &seed),
      seed,
    }
  }
}

impl<T, D, S: SeedExt, B> Bates<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Replace `drift`, all else unchanged; panics on a non-finite rate.
  pub fn with_drift(mut self, drift: Drift<T>) -> Self {
    drift.assert_finite();
    self.drift = drift;
    self
  }

  /// Replace `lambda`, all else unchanged.
  pub fn with_lambda(mut self, lambda: T) -> Self {
    assert!(
      lambda.is_finite(),
      "lambda must satisfy `lambda.is_finite()`, got lambda = {lambda:?}"
    );
    assert!(
      lambda >= T::zero(),
      "lambda must satisfy `lambda >= T::zero()`, got lambda = {lambda:?}"
    );
    self.lambda = lambda;
    self
  }

  /// Replace the jump law, all else unchanged: `k` stays, so set it to the new law's mean.
  pub fn with_jump_law(mut self, law: D) -> Self {
    self.jumps.law = law;
    self
  }

  /// The law of the relative jump `Y − 1`.
  pub fn jump_law(&self) -> &D {
    &self.jumps.law
  }

  /// Replace `k`, all else unchanged.
  pub fn with_k(mut self, k: T) -> Self {
    assert!(
      k.is_finite(),
      "k must satisfy `k.is_finite()`, got k = {k:?}"
    );
    self.k = k;
    self
  }

  /// Replace `alpha`, all else unchanged.
  pub fn with_alpha(mut self, alpha: T) -> Self {
    assert!(
      alpha.is_finite(),
      "alpha must satisfy `alpha.is_finite()`, got alpha = {alpha:?}"
    );
    self.alpha = alpha;
    self
  }

  /// Replace `beta`, all else unchanged.
  pub fn with_beta(mut self, beta: T) -> Self {
    assert!(
      beta.is_finite(),
      "beta must satisfy `beta.is_finite()`, got beta = {beta:?}"
    );
    self.beta = beta;
    self
  }

  /// Replace `sigma`, all else unchanged.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    self.sigma = sigma;
    self
  }

  /// Replace `rho`; rebuilds the cached correlated-Gaussian generator
  /// (`cgns`) so the new correlation actually reaches the sampler instead
  /// of a stale one computed from the old `rho`.
  pub fn with_rho(mut self, rho: T) -> Self {
    assert!(
      rho.is_finite(),
      "rho must satisfy `rho.is_finite()`, got rho = {rho:?}"
    );
    self.rho = rho;
    self.cgns = Cgns::new(rho, self.n - 1, self.t, Unseeded);
    self
  }

  /// Replace `s0`, all else unchanged.
  pub fn with_s0(mut self, s0: T) -> Self {
    assert!(
      s0.is_finite(),
      "s0 must satisfy `s0.is_finite()`, got s0 = {s0:?}"
    );
    self.s0 = s0;
    self
  }

  /// Replace `v0`, all else unchanged.
  pub fn with_v0(mut self, v0: T) -> Self {
    assert!(
      v0.is_finite(),
      "v0 must satisfy `v0.is_finite()`, got v0 = {v0:?}"
    );
    assert!(
      v0 >= T::zero(),
      "v0 must satisfy `v0 >= T::zero()`, got v0 = {v0:?}"
    );
    self.v0 = v0;
    self
  }

  /// Replace `boundary`, all else unchanged.
  pub fn with_boundary(mut self, boundary: VarianceBoundary) -> Self {
    self.boundary = boundary;
    self
  }

  /// Replace `n`; rebuilds the correlated-Gaussian generator, whose length follows `n`.
  pub fn with_points(mut self, n: usize) -> Self {
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    self.n = n;
    self.cgns = Cgns::new(self.rho, n - 1, self.t, Unseeded);
    self
  }

  /// Replace `t`; rebuilds the correlated-Gaussian generator, whose step follows `t`.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    self.t = t;
    self.cgns = Cgns::new(self.rho, self.n - 1, t, Unseeded);
    self
  }

  /// Replace the seed, re-deriving the jumps' seed from it as `new` does.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.jumps.seed = seed.clone().derive();
    self.seed = seed;
    self
  }

  /// The drift net of the `λk` compensator, one expression for the host step and the device launch.
  fn drift_c(&self) -> T {
    self.drift.rate() - self.lambda * self.k
  }
}

impl<T, D, S: SeedExt, B> Bates<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  /// The lognormal law when `D` is it, the one law that takes the log-price step.
  fn log_jumps(&self) -> Option<&LogNormalJump<T>> {
    LogNormalJump::of(&self.jumps.law)
  }

  /// The jump-size law as the Euler engine's kernels draw it, `None` for a
  /// distribution they do not carry.
  fn device_jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    match self.log_jumps() {
      Some(law) => Some(crate::euler::JumpSizes::Normal {
        mean: law.nu(),
        sd: law.omega(),
      }),
      None => crate::process::cpoisson::device_jump_sizes(&self.jumps.law)
        .and_then(crate::euler::JumpSizes::product),
    }
  }
}

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for Bates<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  /// The lognormal law launches the log-price families, any other law the price ones; the variance
  /// keeps whichever treatment of the boundary the process was built with.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    use crate::euler::EulerSpec;
    let (drift_c, alpha, beta, sigma, rho) =
      (self.drift_c(), self.alpha, self.beta, self.sigma, self.rho);
    match (self.log_jumps().is_some(), self.boundary) {
      (true, VarianceBoundary::Floor) => EulerSpec::BatesJump {
        drift_c,
        alpha,
        beta,
        sigma,
        rho,
      },
      (true, VarianceBoundary::Reflect) => EulerSpec::BatesJumpReflected {
        drift_c,
        alpha,
        beta,
        sigma,
        rho,
      },
      (false, VarianceBoundary::Floor) => EulerSpec::Bates1996 {
        drift_c,
        alpha,
        beta,
        sigma,
        rho,
      },
      (false, VarianceBoundary::Reflect) => EulerSpec::Bates1996Reflected {
        drift_c,
        alpha,
        beta,
        sigma,
        rho,
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

  fn time_step(&self) -> T {
    self.cgns.dt()
  }

  /// The jump intensity per unit time; `None` when the process carries none,
  /// which is what makes a zero-intensity build skip the draw entirely.
  fn jump_intensity(&self) -> Option<T> {
    (self.lambda > T::zero()).then_some(self.lambda)
  }

  /// The size law as the kernels draw it. A law they do not carry never
  /// reaches a launch — [`ProcessExt::sample`] keeps such a process on the
  /// host — so asking for one here is a caller bypassing that guard.
  fn jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    if self.lambda <= T::zero() {
      return None;
    }
    Some(self.device_jump_sizes().expect(
      "Bates: the jump-size distribution is not a law the Euler engine compounds multiplicatively on a device; \
       sample through `ProcessExt`, which keeps it on the host",
    ))
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

backend_switch!([T, D, S: SeedExt] Bates<T, D, S> { drift, lambda, k, alpha, beta, sigma, rho, n, s0, v0, t, boundary, cgns, jumps, seed } via euler where  T: FloatExt,  D: Distribution<T> + Send + Sync);

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed for Bates<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
}

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Bates<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = BatesSampler<'s, T, D, S>
  where
    Self: 's;

  /// Chunk-local children of the diffusion's and the jumps' seeds; the shared ones would let
  /// concurrent chunks race, and `cgns`'s own `Unseeded` seed is never read.
  fn sampler(&self) -> BatesSampler<'_, T, D, S> {
    BatesSampler {
      n: self.n,
      s0: self.s0,
      v0: self.v0.max(T::zero()),
      lambda: self.lambda,
      alpha: self.alpha,
      beta: self.beta,
      sigma: self.sigma,
      drift_c: self.drift_c(),
      boundary: self.boundary,
      dt: self.cgns.dt(),
      cgns: self.cgns,
      jump_distribution: &self.jumps.law,
      log_jumps: self.log_jumps(),
      jump_seed: self.jumps.seed.derive(),
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine when the jump-size law is one the device
  /// kernels draw; any other law keeps the process on the host, chunked
  /// exactly as [`ProcessExt`] chunks.
  fn sample(&self) -> [Array1<T>; 2] {
    if self.device_ready() {
      self.backend.system_sample(self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&[Array1<T>; 2]) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      self.backend.system_paths_map(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<[Array1<T>; 2]> {
    if self.device_ready() {
      self.backend.system_paths(self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<[Array1<T>; 2], crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_system_sample(self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<[Array1<T>; 2]>, crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_system_paths(self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// What a device runs here: it has no jumps, or its sizes
  /// follow a law the kernels draw. Anything else samples on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.lambda <= T::zero() || self.device_jump_sizes().is_some()))
      .then_some("a jump-size law the kernels do not draw")
  }
}

/// Reusable [`Bates`] sampling state: the correlated-Gaussian generator, the borrowed jump law and
/// two derived seeds, one for the diffusion and one for the jump arrivals.
#[doc(hidden)]
pub struct BatesSampler<'a, T, D, S: SeedExt>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  n: usize,
  s0: T,
  v0: T,
  lambda: T,
  alpha: T,
  beta: T,
  sigma: T,
  drift_c: T,
  boundary: VarianceBoundary,
  dt: T,
  cgns: Cgns<T>,
  jump_distribution: &'a D,
  log_jumps: Option<&'a LogNormalJump<T>>,
  jump_seed: S,
  seed: S,
}

impl<T, D, S: SeedExt> BatesSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  fn fill_paths(&mut self, s: &mut [T], v: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;
    let [cgn1, cgn2] = &self.cgns.sample_impl(&self.seed);
    let jumps = match self.log_jumps {
      Some(law) => law.grid_log_jumps(self.lambda, &self.jump_seed, self.n, dt),
      None => crate::process::cpoisson::grid_relative_increments(
        self.jump_distribution,
        self.lambda,
        &self.jump_seed,
        self.n,
        dt,
      ),
    };

    s[0] = self.s0;
    v[0] = self.v0;

    for i in 1..self.n {
      let v_prev = v[i - 1].max(T::zero());
      s[i] = match self.log_jumps {
        Some(_) => {
          let half_v = T::from_f64_fast(0.5) * v_prev;
          s[i - 1] * ((self.drift_c - half_v) * dt + v_prev.sqrt() * cgn1[i - 1] + jumps[i]).exp()
        }
        None => {
          s[i - 1]
            + self.drift_c * s[i - 1] * dt
            + s[i - 1] * v_prev.sqrt() * cgn1[i - 1]
            + s[i - 1] * jumps[i]
        }
      };

      let dv = (self.alpha - self.beta * v_prev) * dt + self.sigma * v_prev.sqrt() * cgn2[i - 1];

      v[i] = match self.boundary {
        VarianceBoundary::Floor => (v[i - 1] + dv).max(T::zero()),
        VarianceBoundary::Reflect => (v[i - 1] + dv).abs(),
      }
    }
  }
}

impl<T, D, S: SeedExt> crate::traits::Sealed for BatesSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
}

impl<T, D, S: SeedExt> PathSampler<T> for BatesSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, v] = out;
    self.fill_paths(
      s.as_slice_mut().expect("Bates output must be contiguous"),
      v.as_slice_mut().expect("Bates output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut s = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);
    self.fill_paths(
      s.as_slice_mut().expect("contiguous"),
      v.as_slice_mut().expect("contiguous"),
    );
    [s, v]
  }
}

mod lognormal;

pub use lognormal::LogNormalJump;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyBates;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyBatesSvj;

#[cfg(test)]
mod tests;
