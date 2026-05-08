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
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
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

impl<T: FloatExt> HestonLog<T> {
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
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> HestonLog<T, Deterministic> {
  pub fn seeded(
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
    seed: u64,
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
      seed: Deterministic::new(seed),
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

  fn sample(&self) -> Self::Output {
    let mut s = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return [s, v];
    }

    let s0 = self.s0.unwrap_or(T::one());
    assert!(s0 > T::zero(), "s0 must be > 0 for log-price simulation");
    s[0] = s0;

    v[0] = self.v0.unwrap_or(self.theta).max(T::zero());
    if self.n == 1 {
      return [s, v];
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    let mut dws = vec![T::zero(); n_increments];
    let mut z = vec![T::zero(); n_increments];
    let mut dwv = vec![T::zero(); n_increments];
    let n1 = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &self.seed);
    let n2 = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &self.seed);
    n1.fill_slice_fast(&mut dws);
    n2.fill_slice_fast(&mut z);
    let corr_scale = (T::one() - self.rho * self.rho).sqrt();
    for i in 0..n_increments {
      dwv[i] = self.rho * dws[i] + corr_scale * z[i];
    }

    let drift = self.drift();
    let half = T::from_f64_fast(0.5);

    for i in 1..self.n {
      let v_prev = if self.use_sym.unwrap_or(false) {
        v[i - 1].abs()
      } else {
        v[i - 1].max(T::zero())
      };

      let sqrt_v = v_prev.sqrt();

      let log_inc = (drift - half * v_prev) * dt + sqrt_v * dws[i - 1];
      s[i] = s[i - 1] * log_inc.exp();

      let dv = self.kappa * (self.theta - v_prev) * dt + self.xi * sqrt_v * dwv[i - 1];
      v[i] = if self.use_sym.unwrap_or(false) {
        (v_prev + dv).abs()
      } else {
        (v_prev + dv).max(T::zero())
      };
    }

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
    );
    let [_s, v] = p.sample();
    assert!(v.iter().all(|x| *x >= 0.0));
  }
}

py_process_2x1d!(PyHestonLog, HestonLog,
  sig: (mu=None, b=None, r=None, r_f=None, *, kappa, theta, xi, rho, n, s0=None, v0=None, t=None, use_sym=None, seed=None, dtype=None),
  params: (mu: Option<f64>, b: Option<f64>, r: Option<f64>, r_f: Option<f64>, kappa: f64, theta: f64, xi: f64, rho: f64, n: usize, s0: Option<f64>, v0: Option<f64>, t: Option<f64>, use_sym: Option<bool>)
);
