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

use crate::stochastic::noise::cgns::CGNS;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct HestonLog<T: FloatExt> {
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
  cgns: CGNS<T>,
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
      cgns: CGNS::new(rho, n - 1, t),
    }
  }

  #[inline]
  fn drift(&self) -> T {
    match (self.r, self.r_f, self.b, self.mu) {
      (Some(r), Some(r_f), _, _) => r - r_f,
      (_, _, Some(b), _) => b,
      (_, _, _, Some(mu)) => mu,
      _ => panic!("one of (r and r_f), b, or mu must be provided"),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for HestonLog<T> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let dt = self.cgns.dt();
    let [dws, dwv] = &self.cgns.sample();

    let mut s = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);

    let s0 = self.s0.unwrap_or(T::one());
    assert!(s0 > T::zero(), "s0 must be > 0 for log-price simulation");
    s[0] = s0;

    v[0] = self.v0.unwrap_or(self.theta).max(T::zero());

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
  sig: (kappa, theta, xi, rho, n, mu=None, b=None, r=None, r_f=None, s0=None, v0=None, t=None, use_sym=None, dtype=None),
  params: (mu: Option<f64>, b: Option<f64>, r: Option<f64>, r_f: Option<f64>, kappa: f64, theta: f64, xi: f64, rho: f64, n: usize, s0: Option<f64>, v0: Option<f64>, t: Option<f64>, use_sym: Option<bool>)
);
