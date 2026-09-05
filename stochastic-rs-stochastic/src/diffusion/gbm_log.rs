//! # Gbm Log
//!
//! $$
//! \ln(S_{t+dt}/S_t) = (\mu - \tfrac12\sigma^2)\,dt + \sigma\sqrt{dt}\,Z,\quad Z\sim\mathcal{N}(0,1)
//! $$
//!
//! Exact log-increment scheme guarantees $S_t > 0$.
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Construction-time validator for log-normal-family drift parametrisations
/// where the user can supply any of `(r, r_f)` (foreign-currency / dividend
/// pair), `b` (cost-of-carry), or `mu` (direct drift). Panics at the API
/// boundary if none are provided — moved here from the v1 deferred panic
/// inside `sample()` so the user gets immediate feedback at construction.
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

#[derive(Clone)]
pub struct GbmLog<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Drift rate
  pub mu: Option<T>,
  /// Cost-of-carry rate
  pub b: Option<T>,
  /// Domestic risk-free interest rate
  pub r: Option<T>,
  /// Foreign risk-free interest rate
  pub r_f: Option<T>,
  /// Volatility
  pub sigma: T,
  /// Number of points sampled along the log-Euler GBM path.
  pub n: usize,
  /// Initial asset price (must be > 0)
  pub s0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on) or [`on_device`](Self::on_device).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> GbmLog<T, S> {
  pub fn new(
    mu: Option<T>,
    b: Option<T>,
    r: Option<T>,
    r_f: Option<T>,
    sigma: T,
    n: usize,
    s0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");
    assert!(sigma >= T::zero(), "sigma must be >= 0");
    validate_drift_args(mu, b, r, r_f, "GbmLog");
    Self {
      backend: Cpu,
      mu,
      b,
      r,
      r_f,
      sigma,
      n,
      s0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> GbmLog<T, S, B> {}

impl<T: FloatExt, S: SeedExt, B> GbmLog<T, S, B> {
  #[inline]
  fn drift(&self) -> T {
    // Construction-time `validate_drift_args` guarantees at least one option
    // is set; this match is total at runtime.
    match (self.r, self.r_f, self.b, self.mu) {
      (Some(r), Some(r_f), _, _) => r - r_f,
      (_, _, Some(b), _) => b,
      (_, _, _, Some(mu)) => mu,
      _ => unreachable!("validate_drift_args ensures at least one of (r+r_f), b, mu is set"),
    }
  }

  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }
}

/// The Euler engine's view of log-stepped GBM: the log drift per step is
/// formed here, so the kernel multiplies by one exponential and no literal.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for GbmLog<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let half = T::from_f64_fast(0.5);
    crate::euler::EulerSpec::LogGeometric {
      drift_ln: (self.drift() - half * self.sigma * self.sigma) * dt,
      sigma: self.sigma,
    }
  }

  fn initial_value(&self) -> T {
    self.s0.unwrap_or(T::one())
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

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] GbmLog<T, S> { mu, b, r, r_f, sigma, n, s0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for GbmLog<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = GbmLogSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> GbmLogSampler<T> {
    let dt = self.dt();
    let drift = self.drift();
    let half = T::from_f64_fast(0.5);
    let drift_ln = (drift - half * self.sigma * self.sigma) * dt;
    GbmLogSampler {
      n: self.n,
      s0: self.s0.unwrap_or(T::one()),
      sigma: self.sigma,
      drift_ln,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }

  /// Through the Euler engine: on a device the recursion runs in the kernel,
  /// on the host devices it is this process's own sampler, chunked exactly as
  /// `ProcessExt` chunks.
  fn sample(&self) -> Array1<T> {
    self.backend.euler_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    self.backend.euler_paths_map(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    self.backend.euler_paths(self, m)
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    self.backend.try_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    self.backend.try_euler_paths(self, m)
  }
}

/// Reusable [`GbmLog`] sampling state: precomputed log-drift and the owned
/// Gaussian source.
#[doc(hidden)]
pub struct GbmLogSampler<T: FloatExt> {
  n: usize,
  s0: T,
  sigma: T,
  drift_ln: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> GbmLogSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    assert!(self.s0 > T::zero(), "s0 must be > 0 for log simulation");
    out[0] = self.s0;
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.normal.fill_slice(tail);
    let mut prev = self.s0;
    for z in tail.iter_mut() {
      let log_inc = self.drift_ln + self.sigma * *z;
      let next = prev * log_inc.exp();
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for GbmLogSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("GbmLog output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn price_stays_positive() {
    let p = GbmLog::new(
      Some(0.05_f64),
      None,
      None,
      None,
      0.2,
      1000,
      Some(100.0),
      Some(1.0),
      Unseeded,
    );
    let s = p.sample();
    assert!(s.iter().all(|x| *x > 0.0));
  }
}

py_process_1d!(PyGbmLog, GbmLog,
  sig: (mu=None, b=None, r=None, r_f=None, *, sigma, n, s0=None, t=None, seed=None, dtype=None),
  params: (mu: Option<f64>, b: Option<f64>, r: Option<f64>, r_f: Option<f64>, sigma: f64, n: usize, s0: Option<f64>, t: Option<f64>),
  device
);
