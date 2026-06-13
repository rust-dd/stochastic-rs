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
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::poisson::SimdPoisson;

use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
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
  type Sampler<'s>
    = MjdLogSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> MjdLogSampler<T> {
    // RNG, the Poisson jump-count driver, the diffusion source and the
    // jump-size source are derived from `self.seed` in the same order as the
    // legacy `sample()`, so the first fill reproduces it bit-for-bit; all
    // owned sources advance on reuse for independent paths.
    let dt = self.dt();
    let sqrt_dt = dt.sqrt();
    let drift = self.drift();
    let kappa_j = self.kappa_j();
    let half = T::from_f64_fast(0.5);
    let drift_ln = (drift - self.lambda * kappa_j - half * self.sigma * self.sigma) * dt;

    let rng = self.seed.rng();

    let pois = if self.lambda > T::zero() {
      Some(SimdPoisson::<u32>::new(
        (self.lambda * dt).to_f64().unwrap(),
        &self.seed,
      ))
    } else {
      None
    };

    let normal = SimdNormal::<T>::new(T::zero(), sqrt_dt, &self.seed);
    let jump_normal = SimdNormal::<T>::new(T::zero(), T::one(), &self.seed);

    MjdLogSampler {
      n: self.n,
      sigma: self.sigma,
      nu: self.nu,
      omega: self.omega,
      s0: self.s0.unwrap_or(T::one()),
      drift_ln,
      rng,
      pois,
      normal,
      jump_normal,
    }
  }
}

/// Reusable [`MjdLog`] sampling state: owns the jump-count RNG, the Poisson
/// driver and both Gaussian sources (diffusion and jump-size) so a
/// Monte-Carlo loop pays their setup once.
#[doc(hidden)]
pub struct MjdLogSampler<T: FloatExt> {
  n: usize,
  sigma: T,
  nu: T,
  omega: T,
  s0: T,
  drift_ln: T,
  rng: SimdRng,
  pois: Option<SimdPoisson<u32>>,
  normal: SimdNormal<T>,
  jump_normal: SimdNormal<T>,
}

impl<T: FloatExt> MjdLogSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }

    let s0 = self.s0;
    assert!(s0 > T::zero(), "s0 must be > 0 for log simulation");
    out[0] = s0;
    if out.len() == 1 {
      return;
    }

    let drift_ln = self.drift_ln;

    let mut prev = s0;
    let tail = &mut out[1..];
    self.normal.fill_slice_fast(tail);

    for z in tail.iter_mut() {
      let diff = self.sigma * *z;

      let mut jump_sum = T::zero();
      if let Some(pois) = &self.pois {
        let k: u32 = pois.sample(&mut self.rng);
        if k > 0 {
          let kf = T::from_usize_(k as usize);
          let mut z0 = [T::zero(); 1];
          self.jump_normal.fill_slice_fast(&mut z0);
          jump_sum = self.nu * kf + self.omega * kf.sqrt() * z0[0];
        }
      }

      let log_inc = drift_ln + diff + jump_sum;
      let next = prev * log_inc.exp();
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for MjdLogSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(
      out
        .as_slice_mut()
        .expect("MjdLog output must be contiguous"),
    );
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
