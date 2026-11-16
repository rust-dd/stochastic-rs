//! # Bates SVJ
//!
//! $$
//! \begin{aligned}
//! d\ln S_t &= (\mu - \lambda\kappa_J - \tfrac12 v_t)\,dt + \sqrt{v_t}\,dW_t + Z\,dN_t \\
//! dv_t &= (\alpha - \beta\,v_t)\,dt + \sigma\sqrt{v_t}\,dW_t^v
//! \end{aligned}
//! $$
//!
//! where $\langle dW, dW^v\rangle = \rho\,dt$, $N_t$ is Poisson with intensity $\lambda$,
//! and $Z\sim\mathcal{N}(\nu,\omega^2)$.
//!
//! Log-price reparametrization of Bates D. S. (1996) — *Jumps and
//! Stochastic Volatility: Exchange Rate Processes Implicit in Deutsche
//! Mark Options*, Review of Financial Studies 9(1), 69–107,
//! DOI: 10.1093/rfs/9.1.69, using the lognormal jump-size convention of
//! Merton R. C. (1976) — *Option Pricing When Underlying Stock Returns
//! Are Discontinuous*, Journal of Financial Economics 3(1-2), 125–144,
//! DOI: 10.1016/0304-405X(76)90022-2.
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::poisson::SimdPoisson;

use crate::device::Cpu;
use crate::noise::cgns::Cgns;
use crate::params::Drift;
use crate::params::VarianceBoundary;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Every field has a matching `with_*` builder setter, e.g.
/// `BatesSvj::new(..).with_lambda(0.8).with_rho(-0.4)`.
#[derive(Clone)]
pub struct BatesSvj<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// The asset's drift before the jump compensator: `μ`, a cost of carry, or `r − r_f`.
  pub drift: Drift<T>,
  /// Jump intensity (Poisson arrival rate)
  pub lambda: T,
  /// Mean of the jump log-size Z ~ N(nu, omega^2)
  pub nu: T,
  /// Standard deviation of the jump log-size Z
  pub omega: T,
  /// Variance drift level (often kappa * theta in Heston notation)
  pub alpha: T,
  /// Variance mean-reversion speed (often kappa)
  pub beta: T,
  /// Volatility of variance (vol-of-vol)
  pub sigma: T,
  /// Correlation between asset and variance Brownian motions
  pub rho: T,
  /// Number of discrete time steps in the simulation grid
  pub n: usize,
  /// Initial asset price (must be > 0 for log-price simulation)
  pub s0: Option<T>,
  /// Initial variance level
  pub v0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Treatment of a negative square-root state after a step; the Feller condition is not required.
  pub boundary: VarianceBoundary,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  cgns: Cgns<T>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> BatesSvj<T, S> {
  pub fn new(
    drift: Drift<T>,
    lambda: T,
    nu: T,
    omega: T,
    alpha: T,
    beta: T,
    sigma: T,
    rho: T,
    n: usize,
    s0: Option<T>,
    v0: Option<T>,
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
      nu.is_finite(),
      "nu must satisfy `nu.is_finite()`, got nu = {nu:?}"
    );
    assert!(
      omega.is_finite(),
      "omega must satisfy `omega.is_finite()`, got omega = {omega:?}"
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
    assert!(
      s0.is_none_or(|v| v.is_finite()),
      "s0 must satisfy `s0.is_none_or(|v| v.is_finite())`, got s0 = {s0:?}"
    );
    assert!(
      v0.is_none_or(|v| v.is_finite()),
      "v0 must satisfy `v0.is_none_or(|v| v.is_finite())`, got v0 = {v0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      rho >= -T::one() && rho <= T::one(),
      "rho must satisfy `rho >= -T::one() && rho <= T::one()`, got rho = {rho:?}"
    );
    assert!(
      omega >= T::zero(),
      "omega must satisfy `omega >= T::zero()`, got omega = {omega:?}"
    );
    assert!(
      lambda >= T::zero(),
      "lambda must satisfy `lambda >= T::zero()`, got lambda = {lambda:?}"
    );

    Self {
      backend: Cpu,
      drift,
      lambda,
      nu,
      omega,
      alpha,
      beta,
      sigma,
      rho,
      n,
      s0,
      v0,
      t,
      boundary,
      seed,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> BatesSvj<T, S, B> {
  /// Replace `drift`, all else unchanged; panics on a non-finite rate.
  pub fn with_drift(mut self, drift: Drift<T>) -> Self {
    drift.assert_finite();
    self.drift = drift;
    self
  }

  /// Replace `lambda`, all else unchanged.
  pub fn with_lambda(mut self, lambda: T) -> Self {
    assert!(
      lambda >= T::zero(),
      "lambda must satisfy `lambda >= T::zero()`, got lambda = {lambda:?}"
    );
    self.lambda = lambda;
    self
  }

  /// Replace `nu`, all else unchanged.
  pub fn with_nu(mut self, nu: T) -> Self {
    self.nu = nu;
    self
  }

  /// Replace `omega`, all else unchanged.
  pub fn with_omega(mut self, omega: T) -> Self {
    assert!(
      omega >= T::zero(),
      "omega must satisfy `omega >= T::zero()`, got omega = {omega:?}"
    );
    self.omega = omega;
    self
  }

  /// Replace `alpha`, all else unchanged.
  pub fn with_alpha(mut self, alpha: T) -> Self {
    self.alpha = alpha;
    self
  }

  /// Replace `beta`, all else unchanged.
  pub fn with_beta(mut self, beta: T) -> Self {
    self.beta = beta;
    self
  }

  /// Replace `sigma`, all else unchanged.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    self.sigma = sigma;
    self
  }

  /// Replace `rho`; rebuilds the cached correlated-Gaussian generator
  /// (`cgns`) so the new correlation actually reaches the sampler instead
  /// of a stale one computed from the old `rho`.
  pub fn with_rho(mut self, rho: T) -> Self {
    assert!(
      rho >= -T::one() && rho <= T::one(),
      "rho must satisfy `rho >= -T::one() && rho <= T::one()`, got rho = {rho:?}"
    );
    self.rho = rho;
    self.cgns = Cgns::new(rho, self.n - 1, self.t, Unseeded);
    self
  }

  /// Replace `s0`, all else unchanged.
  pub fn with_s0(mut self, s0: Option<T>) -> Self {
    self.s0 = s0;
    self
  }

  /// Replace `v0`, all else unchanged.
  pub fn with_v0(mut self, v0: Option<T>) -> Self {
    self.v0 = v0;
    self
  }

  /// Replace `boundary`, all else unchanged.
  pub fn with_boundary(mut self, boundary: VarianceBoundary) -> Self {
    self.boundary = boundary;
    self
  }

  /// Replace the number of simulation steps `n`; rebuilds the cached
  /// correlated-Gaussian generator, whose length and step size derive
  /// from `n`.
  pub fn with_steps(mut self, n: usize) -> Self {
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
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

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

impl<T: FloatExt, S: SeedExt, B> BatesSvj<T, S, B> {
  #[inline]
  fn kappa_j(&self) -> T {
    (self.nu + T::from_f64_fast(0.5) * self.omega * self.omega).exp() - T::one()
  }
}

/// The Euler engine's view of the Bates model. The jump compensator is
/// folded into the drift here, as the host sampler folds it, and the
/// intensity travels as a launch argument so the kernel can draw the count
/// once per step.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for BatesSvj<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let drift_c = self.drift.rate() - self.lambda * self.kappa_j();
    match self.boundary {
      VarianceBoundary::Floor => crate::euler::EulerSpec::BatesJump {
        drift_c,
        alpha: self.alpha,
        beta: self.beta,
        sigma: self.sigma,
        rho: self.rho,
      },
      VarianceBoundary::Reflect => crate::euler::EulerSpec::BatesJumpReflected {
        drift_c,
        alpha: self.alpha,
        beta: self.beta,
        sigma: self.sigma,
        rho: self.rho,
      },
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [
      self.s0.unwrap_or(T::one()),
      self.v0.unwrap_or(T::zero()).max(T::zero()),
      T::zero(),
      T::zero(),
    ]
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

  fn jump_intensity(&self) -> Option<T> {
    (self.lambda > T::zero()).then_some(self.lambda)
  }

  /// Lognormal jump sizes, normal in the log-price the family steps.
  fn jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    Some(crate::euler::JumpSizes::Normal {
      mean: self.nu,
      sd: self.omega,
    })
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

backend_switch!([T: FloatExt, S: SeedExt] BatesSvj<T, S> { drift, lambda, nu, omega, alpha, beta, sigma, rho, n, s0, v0, t, boundary, seed, cgns } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for BatesSvj<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for BatesSvj<T, S, B>
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = BatesSvjSampler<T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart.
  fn sampler(&self) -> BatesSvjSampler<T, S> {
    BatesSvjSampler {
      n: self.n,
      s0: self.s0.unwrap_or(T::one()),
      v0: self.v0.unwrap_or(T::zero()).max(T::zero()),
      lambda: self.lambda,
      nu: self.nu,
      omega: self.omega,
      alpha: self.alpha,
      beta: self.beta,
      sigma: self.sigma,
      drift: self.drift.rate(),
      kappa_j: self.kappa_j(),
      dt: self.cgns.dt(),
      boundary: self.boundary,
      cgns: self.cgns,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine: on a device the variance, the diffusion and
  /// the jump all happen in the kernel, on the host devices it is this
  /// process's own sampler, chunked exactly as `ProcessExt` chunks.
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

/// Reusable [`BatesSvj`] sampling state: owns the correlated-Gaussian generator
/// and the seed source. The jump (Poisson + Normal) and asset/variance Brownian
/// generators are rebuilt per fill in the legacy seed-consumption order, so the
/// first call reproduces the original stream bit-for-bit.
#[doc(hidden)]
pub struct BatesSvjSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  s0: T,
  v0: T,
  lambda: T,
  nu: T,
  omega: T,
  alpha: T,
  beta: T,
  sigma: T,
  drift: T,
  kappa_j: T,
  dt: T,
  boundary: VarianceBoundary,
  cgns: Cgns<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> BatesSvjSampler<T, S> {
  fn fill_paths(&mut self, s: &mut [T], v: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;
    let [cgn1, cgn2] = &self.cgns.sample_impl(&self.seed);

    assert!(
      self.s0 > T::zero(),
      "s0 must be > 0 for log-price simulation"
    );
    s[0] = self.s0;
    v[0] = self.v0;

    let drift = self.drift;
    let kappa_j = self.kappa_j;

    let mut z_std = SimdNormal::<f64>::new(0.0, 1.0).seeded(&self.seed);
    // One seed is skipped: the pinned streams that follow take theirs after it.
    self.seed.next_seed();

    let mut pois = if self.lambda > T::zero() {
      Some(SimdPoisson::<u32>::new((self.lambda * dt).to_f64().unwrap()).seeded(&self.seed))
    } else {
      None
    };

    for i in 1..self.n {
      let v_prev = match self.boundary {
        VarianceBoundary::Floor => v[i - 1].max(T::zero()),
        VarianceBoundary::Reflect => v[i - 1].abs(),
      };
      let sqrt_v = v_prev.sqrt();

      let mut jump_sum_z = T::zero();
      if let Some(pois) = &mut pois {
        let k = pois.sample();
        if k > 0 {
          let kf = T::from_usize_(k as usize);
          let z0: f64 = z_std.sample();
          jump_sum_z = self.nu * kf + self.omega * kf.sqrt() * T::from_f64_fast(z0);
        }
      }

      let log_inc = (drift - self.lambda * kappa_j - T::from_f64_fast(0.5) * v_prev) * dt
        + sqrt_v * cgn1[i - 1]
        + jump_sum_z;
      s[i] = s[i - 1] * log_inc.exp();

      let dv = (self.alpha - self.beta * v_prev) * dt + self.sigma * sqrt_v * cgn2[i - 1];
      v[i] = match self.boundary {
        VarianceBoundary::Floor => (v_prev + dv).max(T::zero()),
        VarianceBoundary::Reflect => (v_prev + dv).abs(),
      };
    }
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for BatesSvjSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for BatesSvjSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, v] = out;
    self.fill_paths(
      s.as_slice_mut()
        .expect("BatesSvj output must be contiguous"),
      v.as_slice_mut()
        .expect("BatesSvj output must be contiguous"),
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

// Split out to keep this file under the project's 600-line cap (this type
// now carries a full set of `with_*` builder setters on top of the model
// itself). Same pattern as
// `stochastic-rs-quant/src/pricing/malliavin_thalmaier/engine/delta.rs`.
#[cfg(test)]
#[path = "bates_svj_tests.rs"]
mod tests;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyBatesSvj;
