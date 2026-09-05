//! # Heston Log
//!
//! $$
//! \begin{aligned}
//! d\ln S_t &= (\mu - \tfrac12 v_t)\,dt + \sqrt{v_t}\,dW_t^S \\
//! dv_t &= \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v
//! \end{aligned}
//! $$
//!
//! where $\langle dW^S, dW^v\rangle = \rho\,dt$.
//! Log-spot simulation guarantees $S_t > 0$.
//!
//! Reference: Heston S. L. (1993) — *A Closed-Form Solution for Options
//! with Stochastic Volatility with Applications to Bond and Currency
//! Options*, Review of Financial Studies 6(2), 327–343,
//! DOI: 10.1093/rfs/6.2.327.
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Construction-time validator for drift parametrisations. Panics at the
/// API boundary if none of `(r and r_f)`, `b`, or `mu` is provided.
#[inline]
fn validate_drift_args<T: FloatExt>(
  mu: Option<T>,
  b: Option<T>,
  r: Option<T>,
  r_f: Option<T>,
  type_name: &'static str,
) {
  let has_r_pair = r.is_some() && r_f.is_some();
  if !(has_r_pair || b.is_some() || mu.is_some()) {
    panic!("{type_name}: one of (r and r_f), b, or mu must be provided");
  }
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `HestonLog::new(..).with_kappa(2.0).with_rho(-0.4)`.
pub struct HestonLog<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Drift rate of the asset price
  pub mu: Option<T>,
  /// Cost-of-carry rate
  pub b: Option<T>,
  /// Domestic risk-free interest rate
  pub r: Option<T>,
  /// Foreign risk-free interest rate
  pub r_f: Option<T>,
  /// Variance mean-reversion speed
  pub kappa: T,
  /// Long-run variance level
  pub theta: T,
  /// Volatility of variance (vol-of-vol)
  pub xi: T,
  /// Correlation between asset and variance Brownian motions
  pub rho: T,
  /// Number of discrete time steps
  pub n: usize,
  /// Initial asset price (must be > 0)
  pub s0: Option<T>,
  /// Initial variance level
  pub v0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Use symmetric (abs) instead of truncation (max(0)) for variance
  pub use_sym: Option<bool>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on) or [`on_device`](Self::on_device).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> HestonLog<T, S> {
  pub fn new(
    mu: Option<T>,
    b: Option<T>,
    r: Option<T>,
    r_f: Option<T>,
    kappa: T,
    theta: T,
    xi: T,
    rho: T,
    n: usize,
    s0: Option<T>,
    v0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
    seed: S,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");
    assert!(kappa >= T::zero(), "kappa must be >= 0");
    assert!(theta >= T::zero(), "theta must be >= 0");
    assert!(xi >= T::zero(), "xi must be >= 0");
    assert!(
      rho >= -T::one() && rho <= T::one(),
      "rho must be in [-1, 1]"
    );
    if let Some(v0) = v0 {
      assert!(v0 >= T::zero(), "v0 must be >= 0");
    }
    validate_drift_args(mu, b, r, r_f, "HestonLog");

    Self {
      backend: Cpu,
      mu,
      b,
      r,
      r_f,
      kappa,
      theta,
      xi,
      rho,
      n,
      s0,
      v0,
      t,
      use_sym,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> HestonLog<T, S, B> {
  /// Replace `mu`; re-validates that a drift specification still exists.
  pub fn with_mu(mut self, mu: Option<T>) -> Self {
    self.mu = mu;
    validate_drift_args(self.mu, self.b, self.r, self.r_f, "HestonLog");
    self
  }

  /// Replace `b`; re-validates that a drift specification still exists.
  pub fn with_b(mut self, b: Option<T>) -> Self {
    self.b = b;
    validate_drift_args(self.mu, self.b, self.r, self.r_f, "HestonLog");
    self
  }

  /// Replace `r`; re-validates that a drift specification still exists.
  pub fn with_r(mut self, r: Option<T>) -> Self {
    self.r = r;
    validate_drift_args(self.mu, self.b, self.r, self.r_f, "HestonLog");
    self
  }

  /// Replace `r_f`; re-validates that a drift specification still exists.
  pub fn with_r_f(mut self, r_f: Option<T>) -> Self {
    self.r_f = r_f;
    validate_drift_args(self.mu, self.b, self.r, self.r_f, "HestonLog");
    self
  }

  /// Replace `kappa`, all else unchanged.
  pub fn with_kappa(mut self, kappa: T) -> Self {
    assert!(kappa >= T::zero(), "kappa must be >= 0");
    self.kappa = kappa;
    self
  }

  /// Replace `theta`, all else unchanged.
  pub fn with_theta(mut self, theta: T) -> Self {
    assert!(theta >= T::zero(), "theta must be >= 0");
    self.theta = theta;
    self
  }

  /// Replace `xi`, all else unchanged.
  pub fn with_xi(mut self, xi: T) -> Self {
    assert!(xi >= T::zero(), "xi must be >= 0");
    self.xi = xi;
    self
  }

  /// Replace `rho`, all else unchanged. `HestonLog` has no persisted
  /// correlated-noise cache (`sampler()` builds its Gaussian streams fresh
  /// from `rho` on every call), so unlike `BatesSvj`/`Hkde` this is a plain
  /// field write.
  pub fn with_rho(mut self, rho: T) -> Self {
    assert!(
      rho >= -T::one() && rho <= T::one(),
      "rho must be in [-1, 1]"
    );
    self.rho = rho;
    self
  }

  /// Replace `s0`, all else unchanged.
  pub fn with_s0(mut self, s0: Option<T>) -> Self {
    self.s0 = s0;
    self
  }

  /// Replace `v0`, all else unchanged.
  pub fn with_v0(mut self, v0: Option<T>) -> Self {
    if let Some(v) = v0 {
      assert!(v >= T::zero(), "v0 must be >= 0");
    }
    self.v0 = v0;
    self
  }

  /// Replace `use_sym`, all else unchanged.
  pub fn with_use_sym(mut self, use_sym: Option<bool>) -> Self {
    self.use_sym = use_sym;
    self
  }

  /// Replace the number of simulation steps `n`, all else unchanged.
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
}

impl<T: FloatExt, S: SeedExt, B> HestonLog<T, S, B> {
  #[inline]
  fn drift(&self) -> T {
    // Construction-time `validate_drift_args` guarantees totality at runtime.
    match (self.r, self.r_f, self.b, self.mu) {
      (Some(r), Some(r_f), _, _) => r - r_f,
      (_, _, Some(b), _) => b,
      (_, _, _, Some(mu)) => mu,
      _ => unreachable!("validate_drift_args ensures at least one of (r+r_f), b, mu is set"),
    }
  }
}

/// The Euler engine's view of the log-price Heston model. The drift is
/// resolved once here from whichever of the rate, carry or drift arguments
/// the process was built with.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for HestonLog<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    if self.use_sym.unwrap_or(false) {
      crate::euler::EulerSpec::LogHestonReflected {
        drift: self.drift(),
        kappa: self.kappa,
        theta: self.theta,
        xi: self.xi,
        rho: self.rho,
      }
    } else {
      crate::euler::EulerSpec::LogHeston {
        drift: self.drift(),
        kappa: self.kappa,
        theta: self.theta,
        xi: self.xi,
        rho: self.rho,
      }
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

  fn device_seed(&self) -> u64 {
    rand::Rng::random(&mut self.seed.rng())
  }

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] HestonLog<T, S> { mu, b, r, r_f, kappa, theta, xi, rho, n, s0, v0, t, use_sym, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for HestonLog<T, S, B>
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = HestonLogSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> HestonLogSampler<T> {
    // `saturating_sub(1).max(1)` keeps the noise std finite for n ≤ 1, where
    // the streams are never used; for n ≥ 2 it equals `n - 1`, so the std and
    // hence the derived stream match the legacy `sample` exactly.
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    HestonLogSampler {
      n: self.n,
      s0: self.s0.unwrap_or(T::one()),
      v0: self.v0.unwrap_or(self.theta).max(T::zero()),
      drift: self.drift(),
      kappa: self.kappa,
      theta: self.theta,
      xi: self.xi,
      rho: self.rho,
      dt,
      use_sym: self.use_sym.unwrap_or(false),
      n1: SimdNormal::<T>::new(T::zero(), sqrt_dt, &self.seed),
      n2: SimdNormal::<T>::new(T::zero(), sqrt_dt, &self.seed),
    }
  }

  /// Through the Euler engine: on a device every component steps in the
  /// kernel, on the host devices it is this process's own sampler, chunked
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

  fn try_sample_par(
    &self,
    m: usize,
  ) -> Result<Vec<[Array1<T>; 2]>, crate::device::DeviceError> {
    self.backend.try_system_paths(self, m)
  }
}

/// Reusable [`HestonLog`] sampling state: owns the two Gaussian streams (one
/// driving the asset, one combined into the variance shock) and the
/// precomputed drift / step size so a Monte-Carlo loop reuses both buffers.
#[doc(hidden)]
pub struct HestonLogSampler<T: FloatExt> {
  n: usize,
  s0: T,
  v0: T,
  drift: T,
  kappa: T,
  theta: T,
  xi: T,
  rho: T,
  dt: T,
  use_sym: bool,
  n1: SimdNormal<T>,
  n2: SimdNormal<T>,
}

impl<T: FloatExt> HestonLogSampler<T> {
  fn fill_paths(&mut self, s: &mut [T], v: &mut [T]) {
    if self.n == 0 {
      return;
    }
    assert!(
      self.s0 > T::zero(),
      "s0 must be > 0 for log-price simulation"
    );
    s[0] = self.s0;
    v[0] = self.v0;
    if self.n == 1 {
      return;
    }

    let n_increments = self.n - 1;
    let dt = self.dt;
    let mut dws = vec![T::zero(); n_increments];
    let mut z = vec![T::zero(); n_increments];
    let mut dwv = vec![T::zero(); n_increments];
    self.n1.fill_slice(&mut dws);
    self.n2.fill_slice(&mut z);
    let corr_scale = (T::one() - self.rho * self.rho).sqrt();
    for i in 0..n_increments {
      dwv[i] = self.rho * dws[i] + corr_scale * z[i];
    }

    let drift = self.drift;
    let half = T::from_f64_fast(0.5);

    for i in 1..self.n {
      let v_prev = if self.use_sym {
        v[i - 1].abs()
      } else {
        v[i - 1].max(T::zero())
      };

      let sqrt_v = v_prev.sqrt();

      let log_inc = (drift - half * v_prev) * dt + sqrt_v * dws[i - 1];
      s[i] = s[i - 1] * log_inc.exp();

      let dv = self.kappa * (self.theta - v_prev) * dt + self.xi * sqrt_v * dwv[i - 1];
      v[i] = if self.use_sym {
        (v_prev + dv).abs()
      } else {
        (v_prev + dv).max(T::zero())
      };
    }
  }
}

impl<T: FloatExt> PathSampler<T> for HestonLogSampler<T> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, v] = out;
    self.fill_paths(
      s.as_slice_mut()
        .expect("HestonLog output must be contiguous"),
      v.as_slice_mut()
        .expect("HestonLog output must be contiguous"),
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
mod tests {
  use super::*;

  #[test]
  fn price_stays_positive() {
    let p = HestonLog::new(
      Some(0.05_f64),
      None,
      None,
      None,
      1.5,
      0.04,
      0.3,
      -0.7,
      256,
      Some(100.0),
      Some(0.04),
      Some(1.0),
      Some(false),
      Unseeded,
    );
    let [s, _v] = p.sample();
    assert!(s.iter().all(|x| *x > 0.0));
  }

  #[test]
  fn variance_stays_non_negative() {
    let p = HestonLog::new(
      Some(0.05_f64),
      None,
      None,
      None,
      1.5,
      0.04,
      0.5,
      -0.7,
      256,
      Some(100.0),
      Some(0.04),
      Some(1.0),
      Some(false),
      Unseeded,
    );
    let [_s, v] = p.sample();
    assert!(v.iter().all(|x| *x >= 0.0));
  }
}

py_process_2x1d!(PyHestonLog, HestonLog,
  sig: (mu=None, b=None, r=None, r_f=None, *, kappa, theta, xi, rho, n, s0=None, v0=None, t=None, use_sym=None, seed=None, dtype=None),
  params: (mu: Option<f64>, b: Option<f64>, r: Option<f64>, r_f: Option<f64>, kappa: f64, theta: f64, xi: f64, rho: f64, n: usize, s0: Option<f64>, v0: Option<f64>, t: Option<f64>, use_sym: Option<bool>)
);
