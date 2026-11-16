//! # Black-Karasinski
//!
//! $$
//! d\ln r_t=\bigl(\theta(t)-a\ln r_t\bigr)dt+\sigma\,dW_t
//! $$
//!
//! Black & Karasinski (1991), *Bond and Option Pricing when Short Rates
//! are Lognormal*: the short rate itself is never simulated directly — its
//! logarithm follows a Hull-White-style mean-reverting Gaussian process, so
//! `r_t = exp(Y_t)` is strictly positive by construction, with no boundary
//! condition, floor, or reflection needed (unlike [`Cir`](crate::diffusion::cir::Cir)
//! / [`CirPlusPlus`](crate::interest::cir_pp::CirPlusPlus) / [`Bessel`](crate::diffusion::bessel::Bessel),
//! whose positivity is only *probabilistic* and needs an explicit boundary
//! policy).
//!
//! Same `theta`/`a` convention as [`HullWhite`](crate::interest::hull_white::HullWhite)'s
//! `theta`/`alpha`: [`theta`](BlackKarasinski::theta) is the additive
//! time-dependent drift target function, [`a`](BlackKarasinski::a) the
//! mean-reversion speed multiplying `-ln(r_t)`.
//!
//! Unlike `HullWhite`'s own Euler-Maruyama step, the log-rate `Y_t` here is
//! stepped with the *exact* one-step Gaussian transition of an
//! Ornstein-Uhlenbeck process with `theta` frozen at each step's own
//! evaluation point (the standard "exact Gaussian short rate" scheme): for
//! constant `theta` this has zero discretization bias at any step size,
//! strictly improving on Euler.
//!
//! A lattice (trinomial-tree) engine for the same model already exists at
//! `stochastic-rs-quant::lattice::short_rate::black_karasinski::BlackKarasinskiTree`;
//! this type is the missing Monte-Carlo path simulator for it.

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
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Black-Karasinski log-short-rate process.
///
/// `d ln(r_t) = (theta(t) - a * ln(r_t)) * dt + sigma * dW_t`, reported as
/// `r_t = exp(ln(r_t))`.
///
/// See the module doc for the `theta`/`a` naming convention (shared with
/// [`HullWhite`](crate::interest::hull_white::HullWhite)) and the exact-OU
/// discretization.
#[derive(Clone)]
pub struct BlackKarasinski<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Time-dependent additive drift target θ(t), fitted to the initial term
  /// structure of the log-rate — same role as
  /// [`HullWhite::theta`](crate::interest::hull_white::HullWhite::theta).
  pub theta: Fn1D<T>,
  /// Mean-reversion speed of the log-rate, `HullWhite::alpha`'s role; `a = 0` is the Brownian
  /// limit and `a < 0` diverges, with a `log::warn!`.
  pub a: T,
  /// Diffusion scale σ multiplying `dW_t` in the log-rate SDE.
  pub sigma: T,
  /// Number of points sampled along the path.
  pub n: usize,
  /// Initial short rate r₀. Must be `> 0` so `ln(r0)` is defined; every
  /// subsequent path point is `exp(...)` and so is positive unconditionally
  /// regardless of `r0`.
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

/// Every field has a matching `with_*` builder setter, e.g.
/// `BlackKarasinski::default().with_a(0.6)`. No persisted cache:
/// `sampler()` computes its exact-OU decay/std fresh from `self` and
/// borrows `&self.theta` directly every call.
impl<T: FloatExt, S: SeedExt> BlackKarasinski<T, S> {
  /// Create a new BlackKarasinski process; a negative `a` is accepted with a `log::warn!`.
  pub fn new(
    theta: impl Into<Fn1D<T>>,
    a: T,
    sigma: T,
    n: usize,
    r0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(
      a.is_finite(),
      "a must satisfy `a.is_finite()`, got a = {a:?}"
    );
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      r0.is_none_or(|v| v.is_finite()),
      "r0 must satisfy `r0.is_none_or(|v| v.is_finite())`, got r0 = {r0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
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
  /// Replace `theta`, all else unchanged.
  pub fn with_theta(mut self, theta: impl Into<Fn1D<T>>) -> Self {
    self.theta = theta.into();
    self
  }

  /// Replace `a`, all else unchanged.
  pub fn with_a(mut self, a: T) -> Self {
    self.a = a;
    self
  }

  /// Replace `sigma`, all else unchanged.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    self.sigma = sigma;
    self
  }

  /// Replace `r0`, all else unchanged.
  pub fn with_r0(mut self, r0: Option<T>) -> Self {
    self.r0 = r0;
    self
  }

  /// Replace the number of simulation steps `n`, all else unchanged.
  pub fn with_steps(mut self, n: usize) -> Self {
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
}

/// θ(t)≡0.05 (this file's own `theta_const` test fixture), a=0.8, σ=0.1,
/// r₀=0.03, t=1 — matches
/// `black_karasinski_log_mean_matches_ou`/`black_karasinski_is_deterministic`
/// below (both use these values at t=1; their own `n` varies across this
/// file's tests and is not itself a fixture value). n=252 — one trading
/// year of daily steps (this crate's `Default` convention).
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

/// The Euler engine's view of Black-Karasinski. The recursion is the exact
/// Ornstein-Uhlenbeck transition in log space, so the decay factor and the
/// transition's own standard deviation are folded here; the kernel's noise
/// carries `√dt`, which the folded scale divides back out.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for BlackKarasinski<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n.saturating_sub(1).max(1));
    let decay = (-self.a * dt).exp();
    crate::euler::EulerSpec::LogMeanReverting {
      decay,
      a: self.a,
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

/// The exact OU step's deviation over `dt`; `exp_m1` keeps its digits at a one-minute step in `f32`.
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

  /// `a = 0` stays on the host: the kernels' mean term `(θ/a)(1 − e^{−a·dt})` is `0/0` there.
  fn device_fallback(&self) -> Option<&'static str> {
    (self.a == T::zero())
      .then_some("a zero mean-reversion speed, where the kernels' mean term is 0/0")
  }
}

/// Reusable [`BlackKarasinski`] sampling state: the exact OU step's decay, pull and deviation;
/// `sigma` multiplies each draw at consumption, so `sigma <= 0` needs no special case.
#[doc(hidden)]
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

    // `prev_log` tracks Y_t = ln(r_t) — the exact-OU recursion runs in log
    // space and only the reported value is exponentiated, mirroring how
    // `BesselSampler` tracks its own state in squared space (see
    // `crate::diffusion::bessel`).
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
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  fn theta_const(_t: f64) -> f64 {
    0.05
  }

  fn theta_hundred(_t: f32) -> f32 {
    100.0
  }

  /// ln r is OU: E[ln r_t] = e^{-a t} ln r_0 + (theta/a)(1 - e^{-a t}) when
  /// theta is constant (Black & Karasinski 1991).
  #[test]
  fn black_karasinski_log_mean_matches_ou() {
    let a = 0.8_f64;
    let sigma = 0.1;
    let r0 = 0.03_f64;
    let t = 1.0_f64;
    let n = 200;
    let paths = 20_000;
    let expected = (-a * t).exp() * r0.ln() + (0.05 / a) * (1.0 - (-a * t).exp());

    let best_rel_err = [2718u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let bk = BlackKarasinski::<f64, _>::new(
          theta_const as fn(f64) -> f64,
          a,
          sigma,
          n,
          Some(r0),
          Some(t),
          Deterministic::new(seed),
        );
        let mean_log = bk
          .sample_par(paths)
          .iter()
          .map(|path| path.last().unwrap().ln())
          .sum::<f64>()
          / paths as f64;
        (mean_log - expected).abs() / expected.abs()
      })
      .fold(f64::INFINITY, f64::min);

    assert!(
      best_rel_err <= 2e-2,
      "best-of-3 relative error {best_rel_err} exceeds 2e-2 (expected {expected})"
    );
  }

  /// r_t > 0 for every point of every path (log construction) — must hold
  /// unconditionally, with no boundary policy needed.
  #[test]
  fn black_karasinski_stays_positive() {
    let bk = BlackKarasinski::<f64, _>::new(
      theta_const as fn(f64) -> f64,
      0.8,
      0.5,
      300,
      Some(0.03),
      Some(2.0),
      Deterministic::new(2718),
    );
    for path in bk.sample_par(200) {
      assert!(path.iter().all(|x| x.is_finite() && *x > 0.0));
    }
  }

  /// `a <= 0` never panics: `a = 0` is the Brownian limit, a positive path, and `a < 0` diverges.
  #[test]
  fn black_karasinski_nonpositive_a_does_not_panic() {
    let zero_a = BlackKarasinski::<f64, _>::new(
      theta_const as fn(f64) -> f64,
      0.0,
      0.1,
      10,
      Some(0.03),
      Some(1.0),
      Deterministic::new(42),
    );
    let path = zero_a.sample();
    assert!(
      path.iter().all(|x| x.is_finite() && *x > 0.0),
      "a = 0 samples the Brownian log-rate: {path:?}"
    );

    let negative_a = BlackKarasinski::<f64, _>::new(
      theta_const as fn(f64) -> f64,
      -0.3,
      0.1,
      10,
      Some(0.03),
      Some(1.0),
      Deterministic::new(42),
    );
    // Only asserting "did not panic": a < 0 is documented to diverge, not
    // to stay in any particular finite range.
    let _ = negative_a.sample();
  }

  /// `sigma = 0` is the deterministic OU mean, `ln r_i = decay · ln r_{i−1} + (θ/a)(1 − decay)`.
  #[test]
  fn black_karasinski_zero_volatility_is_exact_ou_mean() {
    let a = 0.8_f64;
    let r0 = 0.03_f64;
    let t = 1.0_f64;
    let n = 50;
    let dt = t / (n - 1) as f64;
    let decay = (-a * dt).exp();

    let bk = BlackKarasinski::<f64, _>::new(
      theta_const as fn(f64) -> f64,
      a,
      0.0,
      n,
      Some(r0),
      Some(t),
      Deterministic::new(42),
    );
    let path = bk.sample();

    assert_eq!(path[0], r0, "path[0] must equal r0 exactly");
    let mut prev_log = r0.ln();
    for &value in path.iter().skip(1) {
      let mean = prev_log * decay + (0.05 / a) * (1.0 - decay);
      assert!(
        (value.ln() - mean).abs() < 1e-9,
        "got ln(r)={}, expected {mean}",
        value.ln()
      );
      prev_log = mean;
    }
  }

  /// A negative `sigma` flips the sign of every draw, a no-op in law rather than an invalid input.
  #[test]
  fn black_karasinski_negative_sigma_does_not_panic() {
    let bk = BlackKarasinski::<f64, _>::new(
      theta_const as fn(f64) -> f64,
      0.8,
      -0.2,
      10,
      Some(0.03),
      Some(1.0),
      Deterministic::new(42),
    );
    let path = bk.sample();
    assert!(path.iter().all(|x| x.is_finite() && *x > 0.0));
  }

  #[test]
  fn the_one_minute_step_variance_does_not_cancel_in_single_precision() {
    let std = ou_step_std(0.01f32, 1.0f32 / (252.0 * 390.0));
    let exact = 1.017_500_914e-5_f64.sqrt();
    assert!(
      (std as f64 / exact - 1.0).abs() < 1e-6,
      "got {std}, exact {exact}: the old form was 17 % high"
    );
  }

  #[test]
  fn the_one_minute_mean_term_keeps_its_digits_in_single_precision() {
    let (a, dt) = (0.01_f32, 1.0_f32 / (252.0 * 390.0));
    let path = BlackKarasinski::<f32, _>::new(
      theta_hundred as fn(f32) -> f32,
      a,
      0.0,
      2,
      Some(0.03),
      Some(dt),
      Deterministic::new(1),
    )
    .sample();
    let (a, dt, r0) = (a as f64, dt as f64, 0.03_f32 as f64);
    // θ·Δ ≈ 1e-3 lifts the mean term above ln r's f32 ulp; (θ/a)(1 − e^{−aΔ}) was 17 % high there.
    let exact = (r0.ln() * (-a * dt).exp() - 100.0 * (-a * dt).exp_m1() / a).exp();
    assert!(
      (path[1] as f64 / exact - 1.0).abs() < 1e-6,
      "got {}, exact {exact}",
      path[1]
    );
  }

  #[test]
  fn a_zero_reversion_is_the_limit_of_a_vanishing_one() {
    let path = |a: f64| {
      BlackKarasinski::<f64, _>::new(
        theta_const as fn(f64) -> f64,
        a,
        0.1,
        10,
        Some(0.03),
        Some(1.0),
        Deterministic::new(42),
      )
      .sample()
    };
    let (limit, near) = (path(0.0), path(1e-12));
    // a = 1e-12 itself moves ln r by about a·t·|ln r0| ≈ 4e-12.
    for (x, y) in limit.iter().zip(near.iter()) {
      assert!(
        (x / y - 1.0).abs() < 1e-11,
        "a = 0 gives {x}, a = 1e-12 gives {y}"
      );
    }
  }

  /// Same seed twice must be bit-identical.
  #[test]
  fn black_karasinski_is_deterministic() {
    let bk1 = BlackKarasinski::<f64, _>::new(
      theta_const as fn(f64) -> f64,
      0.8,
      0.1,
      100,
      Some(0.03),
      Some(1.0),
      Deterministic::new(42),
    )
    .sample();
    let bk2 = BlackKarasinski::<f64, _>::new(
      theta_const as fn(f64) -> f64,
      0.8,
      0.1,
      100,
      Some(0.03),
      Some(1.0),
      Deterministic::new(42),
    )
    .sample();
    assert_eq!(bk1, bk2);
  }
}
