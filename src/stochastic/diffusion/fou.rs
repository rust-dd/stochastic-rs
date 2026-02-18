//! # fOU
//!
//! $$
//! dX_t=\theta(\mu-X_t)\,dt+\sigma\,dB_t^H
//! $$
//!
use ndarray::Array1;

use crate::stochastic::noise::fgn::FGN;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct FOU<T: FloatExt> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Mean-reversion speed.
  pub theta: T,
  /// Long-run mean level.
  pub mu: T,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  fgn: FGN<T>,
}

impl<T: FloatExt> FOU<T> {
  #[must_use]
  pub fn new(hurst: T, theta: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      fgn: FGN::new(hurst, n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for FOU<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.fgn.dt();
    let fgn = self.fgn.sample();

    let mut fou = Array1::<T>::zeros(self.n);
    fou[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      fou[i] = fou[i - 1] + self.theta * (self.mu - fou[i - 1]) * dt + self.sigma * fgn[i - 1];
    }

    fou
  }
}

py_process_1d!(PyFOU, FOU,
  sig: (hurst, theta, mu, sigma, n, x0=None, t=None, dtype=None),
  params: (hurst: f64, theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use super::FOU;
  use crate::traits::ProcessExt;

  #[test]
  #[should_panic(expected = "n must be at least 2")]
  fn fou_requires_at_least_two_points() {
    let _ = FOU::<f64>::new(0.7, 1.0, 0.0, 0.2, 1, Some(0.0), Some(1.0));
  }

  #[test]
  fn fou_sigma_zero_matches_deterministic_euler() {
    let theta = 1.3_f64;
    let mu = 0.8_f64;
    let n = 129_usize;
    let x0 = 0.2_f64;
    let t = 1.0_f64;

    let p = FOU::<f64>::new(0.7, theta, mu, 0.0, n, Some(x0), Some(t));
    let x = p.sample();

    let dt = t / (n as f64 - 1.0);
    let mut expected = x0;
    for i in 1..n {
      expected = expected + theta * (mu - expected) * dt;
      assert!((x[i] - expected).abs() < 1e-12, "mismatch at index {i}");
    }
  }

  #[test]
  fn fou_sample_is_finite() {
    let p = FOU::<f64>::new(0.65, 1.0, 0.0, 0.5, 256, Some(0.1), Some(1.0));
    let x = p.sample();
    assert_eq!(x.len(), 256);
    assert!(x.iter().all(|v| v.is_finite()));
  }
}
