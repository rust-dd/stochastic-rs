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
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

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

pub struct HestonLog<T: FloatExt, S: SeedExt = Unseeded> {
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
  /// Total simulation horizon (defaults to 1)
  pub t: Option<T>,
  /// Use symmetric (abs) instead of truncation (max(0)) for variance
  pub use_sym: Option<bool>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
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

impl<T: FloatExt, S: SeedExt> HestonLog<T, S> {
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

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for HestonLog<T, S> {
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
    self.n1.fill_slice_fast(&mut dws);
    self.n2.fill_slice_fast(&mut z);
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
