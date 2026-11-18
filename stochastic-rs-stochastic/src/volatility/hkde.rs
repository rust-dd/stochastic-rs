//! Heston's variance with Kou's double-exponential jumps in the spot, stepped in the log-spot.
//!
//! $$
//! \begin{aligned}
//! dS_t &= (r-q-\lambda\bar k)S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S + J_t\,S_t\,dN_t \\
//! dv_t &= \kappa(\theta-v_t)\,dt + \sigma_v\sqrt{v_t}\,dW_t^v
//! \end{aligned}
//! $$
//!
//! $J_t$ is Kou's double-exponential jump and $N_t$ a Poisson process of intensity $\lambda$:
//! $$ \ln(1+J) \sim p\,\mathrm{Exp}(\eta_1) + (1-p)\,(-\mathrm{Exp}(\eta_2)) $$
//!
//! Reference: Kirkby, PROJ_Option_Pricing_Matlab
//! Reference: Kou (2002), "A Jump-Diffusion Model for Option Pricing"
//!

use ndarray::Array1;
use rand::RngExt;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::poisson::SimdPoisson;

use crate::device::Cpu;
use crate::noise::cgns::Cgns;
use crate::params::VarianceBoundary;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Heston with Kou double-exponential jumps in the spot.
#[derive(Debug, Clone)]
pub struct Hkde<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Drift rate (or risk-free rate minus dividend yield).
  pub mu: T,
  /// Mean-reversion speed of variance.
  pub kappa: T,
  /// Long-run variance level.
  pub theta: T,
  /// Volatility of variance (vol-of-vol).
  pub sigma_v: T,
  /// Correlation between price and variance Brownian motions.
  pub rho: T,
  /// Initial variance.
  pub v0: T,
  /// Jump intensity (Poisson rate).
  pub lambda: T,
  /// Probability of upward jump.
  pub p_up: T,
  /// Upward jump rate parameter (eta1 > 1 for finite expectation).
  pub eta1: T,
  /// Downward jump rate parameter (eta2 > 0).
  pub eta2: T,
  /// Grid points sampled along the path; `dt = t / (n − 1)`.
  pub n: usize,
  /// Initial spot.
  pub s0: T,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// How a negative square-root state is treated after a step; Feller's condition is not required.
  pub boundary: VarianceBoundary,
  /// Seed strategy.
  pub seed: S,
  cgns: Cgns<T>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Hkde<T, S> {
  /// Panics unless `-1 <= rho <= 1`, `v0 >= 0`, `lambda >= 0`, `eta1 > 1`, `eta2 > 0`,
  /// `s0 > 0`, a given `t` is positive and finite, `n >= 2`, and every parameter is finite.
  pub fn new(
    mu: T,
    kappa: T,
    theta: T,
    sigma_v: T,
    rho: T,
    v0: T,
    lambda: T,
    p_up: T,
    eta1: T,
    eta2: T,
    n: usize,
    s0: T,
    t: Option<T>,
    boundary: VarianceBoundary,
    seed: S,
  ) -> Self {
    Self::check_mu(mu);
    Self::check_kappa(kappa);
    Self::check_theta(theta);
    Self::check_sigma_v(sigma_v);
    Self::check_rho(rho);
    Self::check_v0(v0);
    Self::check_lambda(lambda);
    Self::check_p_up(p_up);
    Self::check_eta1(eta1);
    Self::check_eta2(eta2);
    Self::check_s0(s0);
    check_t(t);
    check_n(n);

    Self {
      backend: Cpu,
      mu,
      kappa,
      theta,
      sigma_v,
      rho,
      v0,
      lambda,
      p_up,
      eta1,
      eta2,
      n,
      s0,
      t,
      boundary,
      seed,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Hkde<T, S, B> {
  fn check_mu(mu: T) {
    require!(mu.is_finite(), mu);
  }

  fn check_kappa(kappa: T) {
    require!(kappa.is_finite(), kappa);
  }

  fn check_theta(theta: T) {
    require!(theta.is_finite(), theta);
  }

  fn check_sigma_v(sigma_v: T) {
    require!(sigma_v.is_finite(), sigma_v);
  }

  fn check_rho(rho: T) {
    require!(rho.is_finite(), rho);
    require!(rho >= -T::one() && rho <= T::one(), rho);
  }

  fn check_v0(v0: T) {
    require!(v0.is_finite(), v0);
    require!(v0 >= T::zero(), v0);
  }

  fn check_lambda(lambda: T) {
    require!(lambda.is_finite(), lambda);
    require!(lambda >= T::zero(), lambda);
  }

  fn check_p_up(p_up: T) {
    require!(p_up.is_finite(), p_up);
  }

  fn check_eta1(eta1: T) {
    require!(eta1.is_finite(), eta1);
    require!(eta1 > T::one(), eta1);
  }

  fn check_eta2(eta2: T) {
    require!(eta2.is_finite(), eta2);
    require!(eta2 > T::zero(), eta2);
  }

  fn check_s0(s0: T) {
    require!(s0.is_finite(), s0);
    require!(s0 > T::zero(), s0);
  }

  /// Replace `mu`; panics unless `mu` is finite.
  pub fn with_mu(mut self, mu: T) -> Self {
    Self::check_mu(mu);
    self.mu = mu;
    self
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

  /// Replace `sigma_v`; panics unless `sigma_v` is finite.
  pub fn with_sigma_v(mut self, sigma_v: T) -> Self {
    Self::check_sigma_v(sigma_v);
    self.sigma_v = sigma_v;
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

  /// Replace `v0`; panics unless `v0 >= 0` and `v0` is finite.
  pub fn with_v0(mut self, v0: T) -> Self {
    Self::check_v0(v0);
    self.v0 = v0;
    self
  }

  /// Replace `lambda`; panics unless `lambda >= 0` and `lambda` is finite.
  pub fn with_lambda(mut self, lambda: T) -> Self {
    Self::check_lambda(lambda);
    self.lambda = lambda;
    self
  }

  /// Replace `p_up`; panics unless `p_up` is finite.
  pub fn with_p_up(mut self, p_up: T) -> Self {
    Self::check_p_up(p_up);
    self.p_up = p_up;
    self
  }

  /// Replace `eta1`; panics unless `eta1 > 1` and `eta1` is finite.
  pub fn with_eta1(mut self, eta1: T) -> Self {
    Self::check_eta1(eta1);
    self.eta1 = eta1;
    self
  }

  /// Replace `eta2`; panics unless `eta2 > 0` and `eta2` is finite.
  pub fn with_eta2(mut self, eta2: T) -> Self {
    Self::check_eta2(eta2);
    self.eta2 = eta2;
    self
  }

  /// Replace `s0`; panics unless `s0 > 0` and `s0` is finite.
  pub fn with_s0(mut self, s0: T) -> Self {
    Self::check_s0(s0);
    self.s0 = s0;
    self
  }

  /// Replace `boundary`.
  pub fn with_boundary(mut self, boundary: VarianceBoundary) -> Self {
    self.boundary = boundary;
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

  /// Replace the seed strategy's value.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }

  /// Kou double-exponential jump compensator: E[e^J - 1]
  #[inline]
  fn k_bar(&self) -> T {
    self.p_up * self.eta1 / (self.eta1 - T::one())
      + (T::one() - self.p_up) * self.eta2 / (self.eta2 + T::one())
      - T::one()
  }
}

/// The Euler engine's view: the jump compensator is folded into the drift, and the sizes are
/// declared, not drawn in the step: Kou's law has no closed-form aggregate, so the kernel sums.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for Hkde<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let drift_c = self.mu - self.lambda * self.k_bar();
    match self.boundary {
      VarianceBoundary::Floor => crate::euler::EulerSpec::KouJumpHeston {
        drift_c,
        kappa: self.kappa,
        theta: self.theta,
        sigma_v: self.sigma_v,
        rho: self.rho,
      },
      VarianceBoundary::Reflect => crate::euler::EulerSpec::KouJumpHestonReflected {
        drift_c,
        kappa: self.kappa,
        theta: self.theta,
        sigma_v: self.sigma_v,
        rho: self.rho,
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

  fn jump_intensity(&self) -> Option<T> {
    (self.lambda > T::zero()).then_some(self.lambda)
  }

  fn jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    Some(crate::euler::JumpSizes::DoubleExponential {
      p_up: self.p_up,
      eta_up: self.eta1,
      eta_down: self.eta2,
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

backend_switch!([T: FloatExt, S: SeedExt] Hkde<T, S> { mu, kappa, theta, sigma_v, rho, v0, lambda, p_up, eta1, eta2, n, s0, t, boundary, seed, cgns } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Hkde<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Hkde<T, S, B> {
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = HkdeSampler<T, S>
  where
    Self: 's;

  /// Derives `self.seed` into the sampler, never clones it: the mixed next tick hash-scrambles
  /// chunk `i`'s basis against chunk `i+1`'s, where a raw snapshot leaves them one stride apart.
  fn sampler(&self) -> HkdeSampler<T, S> {
    HkdeSampler {
      n: self.n,
      mu: self.mu,
      kappa: self.kappa,
      theta: self.theta,
      sigma_v: self.sigma_v,
      v0: self.v0,
      lambda: self.lambda,
      p_up: self.p_up,
      eta1: self.eta1,
      eta2: self.eta2,
      s0: self.s0,
      k_bar: self.k_bar(),
      dt: self.cgns.dt(),
      boundary: self.boundary,
      cgns: self.cgns,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine: a device runs the variance, the diffusion and the jump sum in its
  /// kernel, the host runs this process's own sampler, chunked as `ProcessExt` chunks.
  fn sample(&self) -> [Array1<T>; 2] {
    if self.device_ready() {
      self.backend.system_sample(self)
    } else {
      <Self as crate::euler::EulerSystem<T, 2>>::host_sample(self)
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

  /// A step expecting more than 25 jumps samples on the host, past the kernels' 64-event count.
  fn device_fallback(&self) -> Option<&'static str> {
    crate::euler::count_fallback(
      <Self as crate::euler::EulerSystem<T, 2>>::jump_intensity(self),
      <Self as crate::euler::EulerSystem<T, 2>>::time_step(self),
    )
  }
}

/// Reusable [`Hkde`] sampling state: the correlated-Gaussian generator and the seed source. The
/// Poisson count and the Kou draws are rebuilt per fill, in an order the pinned streams read.
#[doc(hidden)]
#[derive(Debug)]
pub struct HkdeSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  mu: T,
  kappa: T,
  theta: T,
  sigma_v: T,
  v0: T,
  lambda: T,
  p_up: T,
  eta1: T,
  eta2: T,
  s0: T,
  k_bar: T,
  dt: T,
  boundary: VarianceBoundary,
  cgns: Cgns<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> HkdeSampler<T, S> {
  /// Inverse CDF of Exp(η), $F^{-1}(u) = -\ln(1-u)/\eta$, written out so the jump draws stay on
  /// the caller's stream.
  #[inline]
  fn exp_inverse_cdf(u: f64, eta: f64) -> f64 {
    -(1.0 - u).ln() / eta
  }

  /// Sample a single Kou double-exponential jump size (log-jump).
  #[inline]
  fn sample_kou_jump<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> T {
    let u: f64 = rng.random();
    let p = self.p_up.to_f64().unwrap();
    if u < p {
      // Upward jump: Exp(eta1)
      let e = Self::exp_inverse_cdf(rng.random::<f64>(), self.eta1.to_f64().unwrap());
      T::from_f64_fast(e)
    } else {
      // Downward jump: -Exp(eta2)
      let e = Self::exp_inverse_cdf(rng.random::<f64>(), self.eta2.to_f64().unwrap());
      -T::from_f64_fast(e)
    }
  }

  fn fill_paths(&mut self, s: &mut [T], v: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;
    let [cgn1, cgn2] = &self.cgns.sample_impl(&self.seed);

    assert!(
      self.s0 > T::zero(),
      "s0 must satisfy `s0 > T::zero()`, got s0 = {:?}",
      self.s0
    );
    s[0] = self.s0;
    v[0] = self.v0.max(T::zero());

    let k_bar = self.k_bar;
    let mut rng = self.seed.rng();

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

      // Kou jumps
      let mut jump_log = T::zero();
      if let Some(pois) = &mut pois {
        let k = pois.sample();
        for _ in 0..k {
          jump_log += self.sample_kou_jump(&mut rng);
        }
      }

      // Log-price dynamics
      let log_inc = (self.mu - self.lambda * k_bar - T::from_f64_fast(0.5) * v_prev) * dt
        + sqrt_v * cgn1[i - 1]
        + jump_log;
      s[i] = s[i - 1] * log_inc.exp();

      // Variance dynamics (Heston)
      let dv = self.kappa * (self.theta - v_prev) * dt + self.sigma_v * sqrt_v * cgn2[i - 1];
      v[i] = match self.boundary {
        VarianceBoundary::Floor => (v_prev + dv).max(T::zero()),
        VarianceBoundary::Reflect => (v_prev + dv).abs(),
      };
    }
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for HkdeSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for HkdeSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, v] = out;
    self.fill_paths(
      s.as_slice_mut().expect("Hkde output must be contiguous"),
      v.as_slice_mut().expect("Hkde output must be contiguous"),
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

#[cfg(test)]
mod tests;
