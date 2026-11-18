//! Black–Karasinski: a Gaussian log-rate, so `r_t = exp(Y_t)` is positive with no boundary policy.
//!
//! $$
//! d\ln r_t=\bigl(\theta(t)-a\ln r_t\bigr)dt+\sigma\,dW_t
//! $$
//!
//! Its OU transition freezes `theta` over a step: exact for a constant `theta` at any step size.
//!
//! Reference: Black & Karasinski (1991), "Bond and Option Pricing when Short Rates are Lognormal"

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::special::expm1_over;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Black-Karasinski: `d ln r_t = (θ(t) − a ln r_t) dt + σ dW_t`, reported as `r_t = exp(ln r_t)`
/// and stepped by the exact OU transition.
#[derive(Debug, Clone)]
pub struct BlackKarasinski<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Time-dependent drift target θ(t), fitted to the initial term structure of the log-rate, as
  /// [`HullWhite::theta`](crate::interest::hull_white::HullWhite::theta).
  pub theta: Fn1D<T>,
  /// Mean-reversion speed of the log-rate, `HullWhite::alpha`'s role; `a = 0` is the Brownian
  /// limit and `a < 0` diverges, with a `log::warn!`.
  pub a: T,
  /// Diffusion scale σ multiplying `dW_t` in the log-rate SDE.
  pub sigma: T,
  /// Number of points sampled along the path.
  pub n: usize,
  /// Initial short rate r₀, `> 0` so that `ln(r0)` is defined; every later point is an `exp` and
  /// positive whatever `r0`.
  pub r0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or
  /// the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Constant θ(t) ≡ 0.05 used by [`BlackKarasinski`]'s `Default` impl —
/// matches this file's own `theta_const` test fixture.
fn default_theta<T: FloatExt>(_t: T) -> T {
  T::from_f64_fast(0.05)
}

/// Every field has a `with_*` setter (`BlackKarasinski::default().with_a(0.6)`); nothing is
/// cached, since `sampler()` computes the exact-OU decay and deviation from `self` on every call.
impl<T: FloatExt, S: SeedExt> BlackKarasinski<T, S> {
  /// A negative `a` is accepted with a `log::warn!`. Panics unless `n >= 2`, a given `t` is
  /// positive and finite, and every parameter is finite.
  pub fn new(
    theta: impl Into<Fn1D<T>>,
    a: T,
    sigma: T,
    n: usize,
    r0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self::check_a(a);
    Self::check_sigma(sigma);
    check_n(n);
    Self::check_r0(r0);
    check_t(t);
    if a < T::zero() {
      log::warn!("BlackKarasinski::new: a < 0 makes the log-rate diverge, got a = {a:?}");
    }

    Self {
      backend: Cpu,
      theta: theta.into(),
      a,
      sigma,
      n,
      r0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> BlackKarasinski<T, S, B> {
  fn check_a(a: T) {
    require!(a.is_finite(), a);
  }

  fn check_sigma(sigma: T) {
    require!(sigma.is_finite(), sigma);
  }

  fn check_r0(r0: Option<T>) {
    require!(r0.is_none_or(|v| v.is_finite()), r0);
  }

  /// Replace `theta`.
  pub fn with_theta(mut self, theta: impl Into<Fn1D<T>>) -> Self {
    self.theta = theta.into();
    self
  }

  /// Replace `a`; panics unless `a` is finite.
  pub fn with_a(mut self, a: T) -> Self {
    Self::check_a(a);
    self.a = a;
    self
  }

  /// Replace `sigma`; panics unless `sigma` is finite.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    Self::check_sigma(sigma);
    self.sigma = sigma;
    self
  }

  /// Replace `r0`; panics unless `r0` is finite.
  pub fn with_r0(mut self, r0: Option<T>) -> Self {
    Self::check_r0(r0);
    self.r0 = r0;
    self
  }

  /// Replace `n`, the grid points along the path; panics unless `n >= 2`.
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
}

/// θ(t)≡0.05, a=0.8, σ=0.1, r₀=0.03, t=1, the values this file's tests use; 252 points, a trading
/// year of daily closes.
impl<T: FloatExt> Default for BlackKarasinski<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      default_theta::<T> as fn(T) -> T,
      T::from_f64_fast(0.8),
      T::from_f64_fast(0.1),
      252,
      Some(T::from_f64_fast(0.03)),
      Some(T::one()),
      Unseeded,
    )
  }
}

/// The Euler engine's view: the exact OU transition in log space, with the decay and the
/// transition's deviation folded here; the kernel's noise carries `√dt`, which the scale undoes.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for BlackKarasinski<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n.saturating_sub(1).max(1));
    crate::euler::EulerSpec::LogMeanReverting {
      decay: (-self.a * dt).exp(),
      pull: dt * expm1_over(-self.a * dt),
      sigma_eff: self.sigma * ou_step_std(self.a, dt) / dt.sqrt(),
    }
  }

  fn initial_value(&self) -> T {
    self.r0.unwrap_or(T::one()).ln()
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  /// The mean-reversion level at each grid point.
  fn curve(&self) -> Option<Vec<T>> {
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n.saturating_sub(1).max(1));
    Some(
      (0..self.n)
        .map(|i| self.theta.call(T::from_usize_(i) * dt))
        .collect(),
    )
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

/// The exact OU step's deviation over `dt`; `exp_m1` keeps its digits at a one-minute step in
/// `f32`.
pub(crate) fn ou_step_std<T: FloatExt>(a: T, dt: T) -> T {
  (dt * expm1_over(-(a + a) * dt)).sqrt()
}

backend_switch!([T: FloatExt, S: SeedExt] BlackKarasinski<T, S> { theta, a, sigma, n, r0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for BlackKarasinski<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for BlackKarasinski<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = BlackKarasinskiSampler<'s, T>
  where
    Self: 's;

  fn sampler(&self) -> BlackKarasinskiSampler<'_, T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    BlackKarasinskiSampler {
      n: self.n,
      r0: self.r0.unwrap_or(T::one()),
      dt,
      decay: (-self.a * dt).exp(),
      pull: dt * expm1_over(-self.a * dt),
      diff_scale: self.sigma,
      theta: &self.theta,
      normal: SimdNormal::<T>::new(T::zero(), ou_step_std(self.a, dt)).seeded(&self.seed),
    }
  }

  /// Through the Euler engine; a configuration the kernels do not carry samples on the host,
  /// chunked exactly as `ProcessExt` chunks.
  fn sample(&self) -> Array1<T> {
    if self.device_ready() {
      self.backend.euler_sample(self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      self.backend.euler_paths_map(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    if self.device_ready() {
      self.backend.euler_paths_map_view(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, |path| f(path.view()))
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    if self.device_ready() {
      self.backend.euler_paths(self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_sample(self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_euler_paths(self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }
}

/// Reusable [`BlackKarasinski`] sampling state: the exact OU step's decay, pull and deviation;
/// `sigma` multiplies each draw at consumption, so `sigma <= 0` needs no special case.
#[doc(hidden)]
#[derive(Debug)]
pub struct BlackKarasinskiSampler<'a, T: FloatExt> {
  n: usize,
  r0: T,
  dt: T,
  decay: T,
  /// `(1 − e^{−a·dt})/a`, the weight of the frozen `theta` in the step's mean.
  pull: T,
  diff_scale: T,
  theta: &'a Fn1D<T>,
  normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt> BlackKarasinskiSampler<'_, T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.r0;
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.normal.fill_slice(tail);

    // `prev_log` tracks Y_t = ln(r_t): the exact-OU recursion runs in log space and only the
    // reported value is exponentiated.
    let mut prev_log = self.r0.ln();
    for (k, z) in tail.iter_mut().enumerate() {
      let i = k + 1;
      let t_i = T::from_usize_(i) * self.dt;
      let mean = prev_log * self.decay + self.theta.call(t_i) * self.pull;
      let next_log = mean + self.diff_scale * *z;
      *z = next_log.exp();
      prev_log = next_log;
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for BlackKarasinskiSampler<'_, T> {}

impl<T: FloatExt> PathSampler<T> for BlackKarasinskiSampler<'_, T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("BlackKarasinski output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyBlackKarasinski;

#[cfg(test)]
mod tests;
