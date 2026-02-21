//! # MJD Log
//!
//! $$
//! \ln\!\bigl(\tfrac{S_{t+dt}}{S_t}\bigr) = (\mu - \lambda\kappa_J - \tfrac12\sigma^2)\,dt
//!   + \sigma\sqrt{dt}\,Z_1 + \sum_{j=1}^{K} Z_j
//! $$
//!
//! where $K\sim\mathrm{Poisson}(\lambda\,dt)$, $Z_j\sim\mathcal{N}(\nu,\omega^2)$,
//! and $\kappa_J = e^{\nu+\frac12\omega^2}-1$.
//! Log-spot scheme guarantees $S_t > 0$.
//!
use ndarray::Array1;
use rand_distr::Distribution;

use crate::distributions::normal::SimdNormal;
use crate::distributions::poisson::SimdPoisson;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct MJDLog<T: FloatExt> {
  /// Drift rate
  pub mu: Option<T>,
  /// Cost-of-carry rate
  pub b: Option<T>,
  /// Domestic risk-free interest rate
  pub r: Option<T>,
  /// Foreign risk-free interest rate
  pub r_f: Option<T>,
  /// Diffusion volatility
  pub sigma: T,
  /// Jump intensity (Poisson arrival rate)
  pub lambda: T,
  /// Mean of jump log-size Z ~ N(nu, omega^2)
  pub nu: T,
  /// Standard deviation of jump log-size Z
  pub omega: T,
  /// Number of discrete time steps
  pub n: usize,
  /// Initial asset price (must be > 0)
  pub s0: Option<T>,
  /// Total simulation horizon (defaults to 1)
  pub t: Option<T>,
}

impl<T: FloatExt> MJDLog<T> {
  pub fn new(
    mu: Option<T>,
    b: Option<T>,
    r: Option<T>,
    r_f: Option<T>,
    sigma: T,
    lambda: T,
    nu: T,
    omega: T,
    n: usize,
    s0: Option<T>,
    t: Option<T>,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");
    assert!(sigma >= T::zero(), "sigma must be >= 0");
    assert!(lambda >= T::zero(), "lambda must be >= 0");
    assert!(omega >= T::zero(), "omega must be >= 0");
    Self { mu, b, r, r_f, sigma, lambda, nu, omega, n, s0, t }
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

  #[inline]
  fn kappa_j(&self) -> T {
    (self.nu + T::from_f64_fast(0.5) * self.omega * self.omega).exp() - T::one()
  }
}

impl<T: FloatExt> ProcessExt<T> for MJDLog<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.dt();
    let sqrt_dt = dt.sqrt();
    let drift = self.drift();
    let kappa_j = self.kappa_j();
    let half = T::from_f64_fast(0.5);

    let drift_ln = (drift - self.lambda * kappa_j - half * self.sigma * self.sigma) * dt;

    let mut s = Array1::<T>::zeros(self.n);
    let s0 = self.s0.unwrap_or(T::one());
    assert!(s0 > T::zero(), "s0 must be > 0 for log simulation");
    s[0] = s0;

    let mut rng = rand::rng();
    let z_std = SimdNormal::<f64, 64>::new(0.0, 1.0);

    let pois = if self.lambda > T::zero() {
      Some(SimdPoisson::<u32>::new((self.lambda * dt).to_f64().unwrap()))
    } else {
      None
    };

    for i in 1..self.n {
      let z1: f64 = z_std.sample(&mut rng);
      let diff = self.sigma * sqrt_dt * T::from_f64_fast(z1);

      let mut jump_sum = T::zero();
      if let Some(pois) = &pois {
        let k: u32 = pois.sample(&mut rng);
        if k > 0 {
          let kf = T::from_usize_(k as usize);
          let z0: f64 = z_std.sample(&mut rng);
          jump_sum = self.nu * kf + self.omega * kf.sqrt() * T::from_f64_fast(z0);
        }
      }

      let log_inc = drift_ln + diff + jump_sum;
      s[i] = s[i - 1] * log_inc.exp();
    }

    s
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn price_stays_positive() {
    let p = MJDLog::new(
      Some(0.05_f64), None, None, None,
      0.2, 0.5, -0.1, 0.15,
      256, Some(100.0), Some(1.0),
    );
    let s = p.sample();
    assert!(s.iter().all(|x| *x > 0.0));
  }
}

py_process_1d!(PyMJDLog, MJDLog,
  sig: (sigma, lambda_, nu, omega, n, mu=None, b=None, r=None, r_f=None, s0=None, t=None, dtype=None),
  params: (mu: Option<f64>, b: Option<f64>, r: Option<f64>, r_f: Option<f64>, sigma: f64, lambda_: f64, nu: f64, omega: f64, n: usize, s0: Option<f64>, t: Option<f64>)
);
