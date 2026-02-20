use ndarray::Array1;
use rand::Rng;
use rand_distr::Distribution;

use super::clamp_open01;
use crate::distributions::poisson::SimdPoisson;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Classical tempered-stable subordinator approximation.
///
/// Uses truncated-stable large jumps with exponential thinning and
/// deterministic small-jump drift:
/// `nu(dx) = c * exp(-mu x) * x^{-1-alpha} dx`, `x > 0`, `alpha in (0,1)`.
pub struct TemperedStableSubordinator<T: FloatExt> {
  /// Stable index in `(0,1)`.
  pub alpha: T,
  /// Levy density scale.
  pub c: T,
  /// Tempering rate.
  pub mu: T,
  /// Truncation threshold for the large-jump approximation.
  pub epsilon: T,
  /// Number of grid points.
  pub n: usize,
  /// Initial level.
  pub x0: Option<T>,
  /// Horizon.
  pub t: Option<T>,
}

impl<T: FloatExt> TemperedStableSubordinator<T> {
  pub fn new(alpha: T, c: T, mu: T, epsilon: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    assert!(
      alpha > T::zero() && alpha < T::one(),
      "alpha must be in (0,1)"
    );
    assert!(c > T::zero(), "c must be positive");
    assert!(mu > T::zero(), "mu must be positive");
    assert!(epsilon > T::zero(), "epsilon must be positive");
    Self {
      alpha,
      c,
      mu,
      epsilon,
      n,
      x0,
      t,
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for TemperedStableSubordinator<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut out = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return out;
    }
    out[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return out;
    }

    let alpha = self.alpha.to_f64().unwrap();
    let c = self.c.to_f64().unwrap();
    let mu = self.mu.to_f64().unwrap();
    let eps = self.epsilon.to_f64().unwrap();
    let t_max = self.t.unwrap_or(T::one()).to_f64().unwrap();
    let dt = t_max / (self.n - 1) as f64;

    let lambda0 = (c / alpha) * eps.powf(-alpha);
    let poisson = SimdPoisson::<u32>::new(lambda0 * dt);
    let small_jump_drift = dt * c * eps.powf(1.0 - alpha) / (1.0 - alpha);

    let mut rng = rand::rng();
    let mut level = out[0].to_f64().unwrap();
    for i in 1..self.n {
      let n_candidates = poisson.sample(&mut rng) as usize;
      let mut jump_sum = 0.0f64;
      for _ in 0..n_candidates {
        let u = clamp_open01(rng.random::<f64>());
        let x = eps * u.powf(-1.0 / alpha);
        let accept = rng.random::<f64>() <= (-mu * x).exp();
        if accept {
          jump_sum += x;
        }
      }
      level += small_jump_drift + jump_sum;
      out[i] = T::from_f64_fast(level);
    }
    out
  }
}

py_process_1d!(PyTemperedStableSubordinator, TemperedStableSubordinator,
  sig: (alpha, c, mu, epsilon, n, x0=None, t=None, dtype=None),
  params: (alpha: f64, c: f64, mu: f64, epsilon: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
