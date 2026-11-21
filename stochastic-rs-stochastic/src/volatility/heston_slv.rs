//! Heston under a leverage function and a mixing fraction `η` in `[0, 1]`, local volatility at 0.
//!
//! $$
//! \begin{aligned}
//! dS_t &= \mu S_t\,dt + L(t, S_t)\sqrt{V_t}\,S_t\,dW_t^S\\
//! dV_t &= \kappa(\theta - V_t)\,dt + \eta\sigma\sqrt{V_t}\,dW_t^V,
//! \qquad d\langle W^S, W^V\rangle_t = \rho\,dt
//! \end{aligned}
//! $$
//!
//! $L(t, S)$ is read at the step's start; `HestonLog` and quant's calibration run this scheme:
//!
//! $$
//! S_{n+1} = S_n \exp\bigl((\mu - \tfrac12 L^2 V_n^+)\Delta t + L\sqrt{V_n^+}\,\Delta W\bigr),\quad
//! V_{n+1} = \max\bigl(0, V_n + \kappa(\theta - V_n^+)\Delta t +
//! \eta\sigma\sqrt{V_n^+}\,\Delta B\bigr)
//! $$
//!
//! Reference: Jex, Henderson & Wang (1999), Risk 12(11), 72–75
//! Reference: Lipton (2002), Risk 15(2), 61–65
//! Reference: Guyon & Henry-Labordère (2012), Risk 25(1), 88–93
//! Reference: van der Stoep, Grzelak & Oosterlee (2014), IJTAF 17(7), DOI 10.1142/S0219024914500459

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::device::Cpu;
use crate::device::DeviceError;
use crate::params::VarianceBoundary;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::Expr;
use crate::traits::FloatExt;
use crate::traits::Fn2D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Every field has a matching `with_*` builder setter, e.g.
/// `HestonSlv::default().with_eta(0.5).with_leverage(surface)`.
#[derive(Debug, Clone)]
pub struct HestonSlv<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Initial spot.
  pub s0: T,
  /// Initial variance.
  pub v0: T,
  /// Mean-reversion speed κ of the variance.
  pub kappa: T,
  /// Long-run variance θ.
  pub theta: T,
  /// Vol-of-vol σ of the underlying Heston model, before mixing.
  pub sigma: T,
  /// Spot–variance correlation ρ.
  pub rho: T,
  /// Drift μ of the spot.
  pub mu: T,
  /// Mixing fraction η: the vol-of-vol the variance moves under is `η σ`.
  pub eta: T,
  /// Leverage function `L(t, S)`.
  pub leverage: Fn2D<T>,
  /// Number of grid points including `t = 0`.
  pub n: usize,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// [`Unseeded`] or a [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> HestonSlv<T, S> {
  /// Panics unless `s0 > 0`, `v0 >= 0`, `kappa >= 0`, `theta >= 0`, `sigma >= 0`, `-1 <= rho <= 1`,
  /// `eta >= 0`, a given `t` is positive and finite, `n >= 2`, and every parameter is finite.
  pub fn new(
    s0: T,
    v0: T,
    kappa: T,
    theta: T,
    sigma: T,
    rho: T,
    mu: T,
    eta: T,
    leverage: impl Into<Fn2D<T>>,
    n: usize,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self::check_s0(s0);
    Self::check_v0(v0);
    Self::check_kappa(kappa);
    Self::check_theta(theta);
    Self::check_sigma(sigma);
    Self::check_rho(rho);
    Self::check_mu(mu);
    Self::check_eta(eta);
    check_t(t);
    check_n(n);
    Self {
      backend: Cpu,
      s0,
      v0,
      kappa,
      theta,
      sigma,
      rho,
      mu,
      eta,
      leverage: leverage.into(),
      n,
      t,
      seed,
    }
  }
}

/// The crate's Heston (s₀=100, v₀=0.04, κ=2, θ=0.04, σ=0.3, ρ=-0.7, μ=0.05) at `η = 1` with a unit
/// leverage expression, so it runs on a device; t=1, 252 points, a trading year of daily closes.
impl<T: FloatExt> Default for HestonSlv<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(100.0),
      T::from_f64_fast(0.04),
      T::from_f64_fast(2.0),
      T::from_f64_fast(0.04),
      T::from_f64_fast(0.3),
      T::from_f64_fast(-0.7),
      T::from_f64_fast(0.05),
      T::one(),
      Expr::lit(1.0).compile().expect("one constant is one op"),
      252,
      Some(T::one()),
      Unseeded,
    )
  }
}

impl<T: FloatExt, S: SeedExt, B> HestonSlv<T, S, B> {
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

  fn check_rho(rho: T) {
    require!(rho.is_finite(), rho);
    require!((-T::one()..=T::one()).contains(&rho), rho);
  }

  fn check_mu(mu: T) {
    require!(mu.is_finite(), mu);
  }

  fn check_eta(eta: T) {
    require!(eta.is_finite(), eta);
    require!(eta >= T::zero(), eta);
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

  /// Replace `rho`; panics unless `-1 <= rho <= 1` and `rho` is finite.
  pub fn with_rho(mut self, rho: T) -> Self {
    Self::check_rho(rho);
    self.rho = rho;
    self
  }

  /// Replace `mu`; panics unless `mu` is finite.
  pub fn with_mu(mut self, mu: T) -> Self {
    Self::check_mu(mu);
    self.mu = mu;
    self
  }

  /// Replace the mixing fraction `eta`; panics unless `eta >= 0` and `eta` is finite.
  pub fn with_eta(mut self, eta: T) -> Self {
    Self::check_eta(eta);
    self.eta = eta;
    self
  }

  /// Replace the leverage function.
  pub fn with_leverage(mut self, leverage: impl Into<Fn2D<T>>) -> Self {
    self.leverage = leverage.into();
    self
  }

  /// Replace the number of grid points `n`; panics unless `n >= 2`.
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

  /// Replace the seed strategy's value.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }

  /// Time step `Δt = t / (n − 1)`.
  pub fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n.saturating_sub(1).max(1))
  }

  /// The vol-of-vol the variance moves under, `η σ`.
  pub fn sigma_mixed(&self) -> T {
    self.eta * self.sigma
  }

  /// The time the step starting at each grid point sees, `(i − 1) Δt` at point `i`: the launch's
  /// first curve, where the leverage is evaluated, as the host does at the step's start.
  fn step_time_curve(&self) -> Vec<T> {
    let dt = self.dt();
    (0..self.n)
      .map(|i| T::from_usize_(i.saturating_sub(1)) * dt)
      .collect()
  }
}

backend_switch!([T: FloatExt, S: SeedExt] HestonSlv<T, S> { s0, v0, kappa, theta, sigma, rho, mu, eta, leverage, n, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for HestonSlv<T, S, B>
{
  /// [`ProcessExt::sample`] keeps a leverage that is not an expression, which the family cannot
  /// carry, on the host, so a launch that asks here bypassed that guard.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    assert!(
      self.device_ready(),
      "HestonSlv: a launch carries a leverage written as an expression; sample through \
       `ProcessExt`, which keeps a closure or a grid on the host"
    );
    crate::euler::EulerSpec::HestonSlv {
      mu: self.mu,
      kappa: self.kappa,
      theta: self.theta,
      sigma: self.sigma_mixed(),
      rho: self.rho,
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
    self.dt()
  }

  fn curves(&self) -> Option<Vec<Vec<T>>> {
    Some(vec![self.step_time_curve()])
  }

  /// The leverage's program, read by the family as `pv`.
  fn program_spec(&self) -> Option<crate::euler::ProgramSpec<'_>> {
    self
      .leverage
      .program()
      .map(|first| crate::euler::ProgramSpec {
        first,
        second: None,
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

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for HestonSlv<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for HestonSlv<T, S, B>
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = HestonSlvSampler<'s, T>
  where
    Self: 's;

  /// The two Gaussian streams are built in `HestonLog`'s order, each derived from `self.seed`:
  /// the unit-leverage limit is that process path for path.
  fn sampler(&self) -> HestonSlvSampler<'_, T> {
    let dt = self.dt();
    let sqrt_dt = dt.sqrt();
    HestonSlvSampler {
      n: self.n,
      s0: self.s0,
      v0: self.v0.max(T::zero()),
      mu: self.mu,
      kappa: self.kappa,
      theta: self.theta,
      sigma_mixed: self.sigma_mixed(),
      rho: self.rho,
      dt,
      leverage: &self.leverage,
      n1: SimdNormal::<T>::new(T::zero(), sqrt_dt).seeded(&self.seed),
      n2: SimdNormal::<T>::new(T::zero(), sqrt_dt).seeded(&self.seed),
    }
  }

  /// Through the Euler engine when the leverage is an expression; a closure or a grid stays on
  /// the host, chunked as [`ProcessExt`] chunks.
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

  fn try_sample(&self) -> Result<[Array1<T>; 2], DeviceError> {
    if self.device_ready() {
      self.backend.try_system_sample(self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<[Array1<T>; 2]>, DeviceError> {
    if self.device_ready() {
      self.backend.try_system_paths(self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// What a device runs: a leverage written as an [`Expr`], which the kernel interprets at every
  /// step; a Rust closure, a tabulated grid or a Python callable stays on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.leverage.program().is_some()))
      .then_some("a leverage that is a closure, a grid or a Python callable rather than an Expr")
  }
}

/// Reusable [`HestonSlv`] sampling state: borrows the leverage and owns the two Gaussian streams
/// (the spot's, and the one combined into the variance shock) and the step size.
#[doc(hidden)]
#[derive(Debug)]
pub struct HestonSlvSampler<'a, T: FloatExt> {
  n: usize,
  s0: T,
  v0: T,
  mu: T,
  kappa: T,
  theta: T,
  sigma_mixed: T,
  rho: T,
  dt: T,
  leverage: &'a Fn2D<T>,
  n1: Seeded<SimdNormal<T>>,
  n2: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt> HestonSlvSampler<'_, T> {
  fn fill_paths(&mut self, s: &mut [T], v: &mut [T]) {
    if self.n == 0 {
      return;
    }
    s[0] = self.s0;
    v[0] = self.v0;
    if self.n == 1 {
      return;
    }

    let n_increments = self.n - 1;
    let mut dws = vec![T::zero(); n_increments];
    let mut z = vec![T::zero(); n_increments];
    let mut dwv = vec![T::zero(); n_increments];
    self.n1.fill_slice(&mut dws);
    self.n2.fill_slice(&mut z);
    let corr_scale = (T::one() - self.rho * self.rho).sqrt();
    for i in 0..n_increments {
      dwv[i] = self.rho * dws[i] + corr_scale * z[i];
    }

    let dt = self.dt;
    let half = T::from_f64_fast(0.5);
    let boundary = VarianceBoundary::FullTruncation;
    let mut carried = v[0];
    for i in 1..self.n {
      let t_prev = T::from_usize_(i - 1) * dt;
      let v_prev = boundary.observed(carried);
      let sqrt_v = v_prev.sqrt();
      let l = self.leverage.call(t_prev, s[i - 1]);

      let log_inc = (self.mu - half * l * l * v_prev) * dt + l * sqrt_v * dws[i - 1];
      s[i] = s[i - 1] * log_inc.exp();

      let (kappa, theta, sigma) = (self.kappa, self.theta, self.sigma_mixed);
      carried = boundary.square_root_step(carried, kappa, theta, sigma, dt, dwv[i - 1]);
      v[i] = boundary.observed(carried);
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for HestonSlvSampler<'_, T> {}

impl<T: FloatExt> PathSampler<T> for HestonSlvSampler<'_, T> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, v] = out;
    let s = s
      .as_slice_mut()
      .expect("HestonSlv spot output must be contiguous");
    let v = v
      .as_slice_mut()
      .expect("HestonSlv variance output must be contiguous");
    assert_eq!(
      s.len(),
      v.len(),
      "spot and variance outputs must share a length"
    );
    self.fill_paths(s, v);
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut out = [Array1::<T>::zeros(self.n), Array1::<T>::zeros(self.n)];
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
pub use python::PyHestonSlv;
