//! # MJD Log
//!
//! $$
//! \ln\!\bigl(\tfrac{S_{t+dt}}{S_t}\bigr) = (\mu - \lambda\kappa_J - \tfrac12\sigma^2)\,dt
//!   + \sigma\sqrt{dt}\,Z_1 + \sum_{j=1}^{K} Z_j
//!
//! $$
//!
//! where $K\sim\mathrm{Poisson}(\lambda\,dt)$, $Z_j\sim\mathcal{N}(\nu,\omega^2)$,
//! and $\kappa_J = e^{\nu+\frac12\omega^2}-1$.
//! Log-spot scheme guarantees $S_t > 0$.
//!
use ndarray::Array1;
use ndarray::s;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::poisson::SimdPoisson;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

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

pub struct MjdLog<T: FloatExt, S: SeedExt = Unseeded> {
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
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> MjdLog<T, S> {
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
    seed: S,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");
    assert!(sigma >= T::zero(), "sigma must be >= 0");
    assert!(lambda >= T::zero(), "lambda must be >= 0");
    assert!(omega >= T::zero(), "omega must be >= 0");
    validate_drift_args(mu, b, r, r_f, "MjdLog");
    Self {
      mu,
      b,
      r,
      r_f,
      sigma,
      lambda,
      nu,
      omega,
      n,
      s0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> MjdLog<T, S> {
  #[inline]
  fn drift(&self) -> T {
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

  #[inline]
  fn kappa_j(&self) -> T {
    (self.nu + T::from_f64_fast(0.5) * self.omega * self.omega).exp() - T::one()
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for MjdLog<T, S> {
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
    let kappa_j = self.kappa_j();
    let half = T::from_f64_fast(0.5);

    let drift_ln = (drift - self.lambda * kappa_j - half * self.sigma * self.sigma) * dt;

    let mut rng = self.seed.rng();

    let pois = if self.lambda > T::zero() {
      Some(SimdPoisson::<u32>::new(
        (self.lambda * dt).to_f64().unwrap(),
        &Unseeded,
      ))
    } else {
      None
    };

    let mut prev = s0;
    let mut tail_view = s.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("MjdLog output tail must be contiguous");
    let normal = SimdNormal::<T>::new(T::zero(), sqrt_dt, &self.seed);
    normal.fill_slice_fast(tail);

    let jump_normal = SimdNormal::<T>::new(T::zero(), T::one(), &self.seed);

    for z in tail.iter_mut() {
      let diff = self.sigma * *z;

      let mut jump_sum = T::zero();
      if let Some(pois) = &pois {
        let k: u32 = pois.sample(&mut rng);
        if k > 0 {
          let kf = T::from_usize_(k as usize);
          let mut z0 = [T::zero(); 1];
          jump_normal.fill_slice_fast(&mut z0);
          jump_sum = self.nu * kf + self.omega * kf.sqrt() * z0[0];
        }
      }

      let log_inc = drift_ln + diff + jump_sum;
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
    let p = MjdLog::new(
      Some(0.05_f64),
      None,
      None,
      None,
      0.2,
      0.5,
      -0.1,
      0.15,
      256,
      Some(100.0),
      Some(1.0),
      Unseeded,
    );
    let s = p.sample();
    assert!(s.iter().all(|x| *x > 0.0));
  }
}

py_process_1d!(PyMjdLog, MjdLog,
  sig: (mu=None, b=None, r=None, r_f=None, *, sigma, lambda_, nu, omega, n, s0=None, t=None, seed=None, dtype=None),
  params: (mu: Option<f64>, b: Option<f64>, r: Option<f64>, r_f: Option<f64>, sigma: f64, lambda_: f64, nu: f64, omega: f64, n: usize, s0: Option<f64>, t: Option<f64>)
);
