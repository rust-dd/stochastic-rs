//! # GBM Log
//!
//! $$
//! \ln(S_{t+dt}/S_t) = (\mu - \tfrac12\sigma^2)\,dt + \sigma\sqrt{dt}\,Z,\quad Z\sim\mathcal{N}(0,1)
//! $$
//!
//! Exact log-increment scheme guarantees $S_t > 0$.
//!
use ndarray::Array1;
use ndarray::s;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct GBMLog<T: FloatExt> {
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
  /// Number of discrete time steps
  pub n: usize,
  /// Initial asset price (must be > 0)
  pub s0: Option<T>,
  /// Total simulation horizon (defaults to 1)
  pub t: Option<T>,
}

impl<T: FloatExt> GBMLog<T> {
  pub fn new(
    mu: Option<T>,
    b: Option<T>,
    r: Option<T>,
    r_f: Option<T>,
    sigma: T,
    n: usize,
    s0: Option<T>,
    t: Option<T>,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");
    assert!(sigma >= T::zero(), "sigma must be >= 0");
    Self {
      mu,
      b,
      r,
      r_f,
      sigma,
      n,
      s0,
      t,
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

  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }
}

impl<T: FloatExt> ProcessExt<T> for GBMLog<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut s = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return s;
    }

    let s0 = self.s0.unwrap_or(T::one());
    assert!(s0 > T::zero(), "s0 must be > 0 for log simulation");
    s[0] = s0;
    if self.n == 1 {
      return s;
    }

    let dt = self.dt();
    let sqrt_dt = dt.sqrt();
    let drift = self.drift();
    let half = T::from_f64_fast(0.5);
    let drift_ln = (drift - half * self.sigma * self.sigma) * dt;

    let mut prev = s0;
    let mut tail_view = s.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("GBMLog output tail must be contiguous");
    T::fill_standard_normal_scaled_slice(tail, sqrt_dt);

    for z in tail.iter_mut() {
      let log_inc = drift_ln + self.sigma * *z;
      let next = prev * log_inc.exp();
      *z = next;
      prev = next;
    }

    s
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn price_stays_positive() {
    let p = GBMLog::new(
      Some(0.05_f64),
      None,
      None,
      None,
      0.2,
      1000,
      Some(100.0),
      Some(1.0),
    );
    let s = p.sample();
    assert!(s.iter().all(|x| *x > 0.0));
  }
}

py_process_1d!(PyGBMLog, GBMLog,
  sig: (mu=None, b=None, r=None, r_f=None, *, sigma, n, s0=None, t=None, dtype=None),
  params: (mu: Option<f64>, b: Option<f64>, r: Option<f64>, r_f: Option<f64>, sigma: f64, n: usize, s0: Option<f64>, t: Option<f64>)
);
