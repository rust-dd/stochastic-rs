//! # Heston stochastic-local volatility
//!
//! $$
//! \begin{aligned}
//! dS_t &= \mu S_t\,dt + L(t, S_t)\sqrt{V_t}\,S_t\,dW_t^S\\
//! dV_t &= \kappa(\theta - V_t)\,dt + \eta\sigma\sqrt{V_t}\,dW_t^V,
//! \qquad d\langle W^S, W^V\rangle_t = \rho\,dt
//! \end{aligned}
//! $$
//!
//! The Heston model under a leverage function $L(t, S)$ and a mixing
//! fraction $\eta \in [0, 1]$: $\eta = 0$ is a pure local-volatility model
//! with $\sigma_{\text{LV}} = L\sqrt{V}$ deterministic, $\eta = 1$ the Heston
//! dynamics under a leverage correction, and a calibrated $L$ makes the
//! model reproduce the vanilla surface at every $\eta$. The leverage is an
//! [`Fn2D`] of `(t, S)`: written as an [`Expr`] it reaches a device, and a
//! Rust closure, a Python callable or a tabulated
//! [`Grid2D`](crate::traits::Grid2D) — what a calibrated
//! `LeverageSurface` converts into — keeps the process on the host.
//!
//! The log-spot is stepped by Euler–Maruyama and the variance by the crate's
//! Euler scheme with absorption at zero,
//! $S_{n+1} = S_n \exp\bigl((\mu - \tfrac12 L^2 V_n^+)\Delta t + L\sqrt{V_n^+}\,\Delta W\bigr)$,
//! $V_{n+1} = \max\bigl(0, V_n + \kappa(\theta - V_n^+)\Delta t + \eta\sigma\sqrt{V_n^+}\,\Delta B\bigr)$,
//! with $L$ read at the step's start. That is
//! [`HestonLog`](crate::volatility::heston_log::HestonLog)'s recursion, and
//! with $\eta = 1$ and $L \equiv 1$ the two agree path for path at the same
//! seed. It is also the recursion the quant crate's leverage calibration and
//! its Monte Carlo pricer run, so a calibrated surface is valid here.
//!
//! References: Jex, M., Henderson, R. & Wang, D. (1999), *Pricing exotics
//! under the smile*, Risk 12(11), 72–75; Lipton, A. (2002), *The vol smile
//! problem*, Risk 15(2), 61–65; Guyon, J. & Henry-Labordère, P. (2012),
//! *Being particular about calibration*, Risk 25(1), 88–93; van der Stoep,
//! A. W., Grzelak, L. A. & Oosterlee, C. W. (2014), *The Heston
//! stochastic-local volatility model: efficient Monte Carlo simulation*,
//! Int. J. Theor. Appl. Finance 17(7),
//! <https://doi.org/10.1142/S0219024914500459>.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::device::Cpu;
use crate::device::DeviceError;
use crate::traits::Expr;
use crate::traits::FloatExt;
use crate::traits::Fn2D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Every field has a matching `with_*` builder setter, e.g.
/// `HestonSlv::default().with_eta(0.5).with_leverage(surface)`.
#[derive(Clone)]
pub struct HestonSlv<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Initial spot, `> 0` (defaults to 1).
  pub s0: Option<T>,
  /// Initial variance v₀ — a variance, not a volatility (defaults to
  /// `theta`).
  pub v0: Option<T>,
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
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> HestonSlv<T, S> {
  pub fn new(
    s0: Option<T>,
    v0: Option<T>,
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
    assert!(n >= 2, "n must be at least 2");
    assert!(kappa >= T::zero(), "kappa must be non-negative");
    assert!(theta >= T::zero(), "theta must be non-negative");
    assert!(sigma >= T::zero(), "sigma must be non-negative");
    assert!(eta >= T::zero(), "eta must be non-negative");
    assert!(
      (-T::one()..=T::one()).contains(&rho),
      "rho must be a correlation in [-1, 1]"
    );
    if let Some(s0) = s0 {
      assert!(s0 > T::zero(), "s0 must be positive");
    }
    if let Some(v0) = v0 {
      assert!(v0 >= T::zero(), "v0 must be non-negative");
    }
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

/// s₀=100, v₀=0.04, κ=2.0, θ=0.04, σ=0.3, ρ=-0.7, μ=0.05 — the crate's
/// Heston parameterization — under full mixing `η = 1` and unit leverage
/// written as an expression, so the default is the Heston model and runs on
/// a device. t=1, n=252 — one trading year of daily steps.
impl<T: FloatExt> Default for HestonSlv<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      Some(T::from_f64_fast(100.0)),
      Some(T::from_f64_fast(0.04)),
      T::from_f64_fast(2.0),
      T::from_f64_fast(0.04),
      T::from_f64_fast(0.3),
      T::from_f64_fast(-0.7),
      T::from_f64_fast(0.05),
      T::one(),
      Expr::lit(1.0),
      252,
      Some(T::one()),
      Unseeded,
    )
  }
}

impl<T: FloatExt, S: SeedExt, B> HestonSlv<T, S, B> {
  /// Replace `s0`, all else unchanged.
  pub fn with_s0(mut self, s0: Option<T>) -> Self {
    if let Some(s) = s0 {
      assert!(s > T::zero(), "s0 must be positive");
    }
    self.s0 = s0;
    self
  }

  /// Replace `v0`, all else unchanged.
  pub fn with_v0(mut self, v0: Option<T>) -> Self {
    if let Some(v) = v0 {
      assert!(v >= T::zero(), "v0 must be non-negative");
    }
    self.v0 = v0;
    self
  }

  /// Replace `kappa`, all else unchanged.
  pub fn with_kappa(mut self, kappa: T) -> Self {
    assert!(kappa >= T::zero(), "kappa must be non-negative");
    self.kappa = kappa;
    self
  }

  /// Replace `theta`, all else unchanged.
  pub fn with_theta(mut self, theta: T) -> Self {
    assert!(theta >= T::zero(), "theta must be non-negative");
    self.theta = theta;
    self
  }

  /// Replace `sigma`, all else unchanged.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    assert!(sigma >= T::zero(), "sigma must be non-negative");
    self.sigma = sigma;
    self
  }

  /// Replace `rho`, all else unchanged.
  pub fn with_rho(mut self, rho: T) -> Self {
    assert!(
      (-T::one()..=T::one()).contains(&rho),
      "rho must be a correlation in [-1, 1]"
    );
    self.rho = rho;
    self
  }

  /// Replace `mu`, all else unchanged.
  pub fn with_mu(mut self, mu: T) -> Self {
    self.mu = mu;
    self
  }

  /// Replace the mixing fraction `eta`, all else unchanged.
  pub fn with_eta(mut self, eta: T) -> Self {
    assert!(eta >= T::zero(), "eta must be non-negative");
    self.eta = eta;
    self
  }

  /// Replace the leverage function, all else unchanged.
  pub fn with_leverage(mut self, leverage: impl Into<Fn2D<T>>) -> Self {
    self.leverage = leverage.into();
    self
  }

  /// Replace the number of grid points `n`, all else unchanged.
  pub fn with_steps(mut self, n: usize) -> Self {
    assert!(n >= 2, "n must be at least 2");
    self.n = n;
    self
  }

  /// Replace the simulation horizon `t`, all else unchanged.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
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

  /// The time the step starting at each grid point sees, `(i − 1) Δt` at
  /// point `i`: the launch's first curve, which is the `t` the leverage is
  /// evaluated at, exactly as the host evaluates it at the step's start.
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
  /// A configuration the family cannot carry — a leverage that is not an
  /// expression — never reaches a launch: [`ProcessExt::sample`] keeps it on
  /// the host, so asking here is a caller bypassing that guard.
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
    [
      self.s0.unwrap_or(T::one()),
      self.v0.unwrap_or(self.theta).max(T::zero()),
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

  fn time_step(&self) -> T {
    self.dt()
  }

  fn curves(&self) -> Option<Vec<Vec<T>>> {
    Some(vec![self.step_time_curve()])
  }

  /// The leverage's program, read by the family as `pv`.
  fn program_spec(&self) -> Option<crate::euler::ProgramSpec<'_>> {
    self.leverage.program().map(|first| crate::euler::ProgramSpec {
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

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for HestonSlv<T, S, B>
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = HestonSlvSampler<'s, T>
  where
    Self: 's;

  /// The two Gaussian streams are built in the order
  /// [`HestonLog`](crate::volatility::heston_log::HestonLog) builds its own,
  /// each a derivation of `self.seed`, which is what makes the unit-leverage
  /// limit that process path for path.
  fn sampler(&self) -> HestonSlvSampler<'_, T> {
    let dt = self.dt();
    let sqrt_dt = dt.sqrt();
    HestonSlvSampler {
      n: self.n,
      s0: self.s0.unwrap_or(T::one()),
      v0: self.v0.unwrap_or(self.theta).max(T::zero()),
      mu: self.mu,
      kappa: self.kappa,
      theta: self.theta,
      sigma_mixed: self.sigma_mixed(),
      rho: self.rho,
      dt,
      leverage: &self.leverage,
      n1: SimdNormal::<T>::new(T::zero(), sqrt_dt, &self.seed),
      n2: SimdNormal::<T>::new(T::zero(), sqrt_dt, &self.seed),
    }
  }

  /// Through the Euler engine when the leverage is an expression; a closure
  /// or a grid keeps the process on the host, chunked exactly as
  /// [`ProcessExt`] chunks.
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

  /// What a device runs here: a leverage written as an
  /// [`Expr`], which the kernel interprets at every
  /// step. A Rust closure, a tabulated grid or a Python callable keeps the
  /// process on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.leverage.program().is_some()))
      .then_some("a leverage that is a closure, a grid or a Python callable rather than an Expr")
  }
}

/// Reusable [`HestonSlv`] sampling state: borrows the leverage and owns the
/// two Gaussian streams (one driving the spot, one combined into the
/// variance shock) and the precomputed step size, so a Monte Carlo loop pays
/// the setup once.
#[doc(hidden)]
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
  n1: SimdNormal<T>,
  n2: SimdNormal<T>,
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
    for i in 1..self.n {
      let t_prev = T::from_usize_(i - 1) * dt;
      let v_prev = v[i - 1].max(T::zero());
      let sqrt_v = v_prev.sqrt();
      let l = self.leverage.call(t_prev, s[i - 1]);

      let log_inc = (self.mu - half * l * l * v_prev) * dt + l * sqrt_v * dws[i - 1];
      s[i] = s[i - 1] * log_inc.exp();

      let dv = self.kappa * (self.theta - v_prev) * dt + self.sigma_mixed * sqrt_v * dwv[i - 1];
      v[i] = (v_prev + dv).max(T::zero());
    }
  }
}

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
    assert_eq!(s.len(), v.len(), "spot and variance outputs must share a length");
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
pub use python::PyHestonSlv;
