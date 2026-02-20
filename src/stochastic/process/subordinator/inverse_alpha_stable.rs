use ndarray::Array1;
use rand::Rng;

use super::sample_positive_stable;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Inverse alpha-stable subordinator:
/// `E_alpha(t) = inf { u >= 0 : D_alpha(u) > t }`.
pub struct InverseAlphaStableSubordinator<T: FloatExt> {
  /// Stability index in `(0, 1)`.
  pub alpha: T,
  /// Laplace scale of the direct stable subordinator.
  pub c: T,
  /// Number of target time-grid points.
  pub n: usize,
  /// Horizon for target time-grid.
  pub t: Option<T>,
  /// Internal grid size for the direct process `D_alpha(u)`.
  pub u_steps: usize,
  /// Optional upper bound for inverse-domain `u`.
  pub u_max: Option<T>,
}

impl<T: FloatExt> InverseAlphaStableSubordinator<T> {
  pub fn new(alpha: T, c: T, n: usize, t: Option<T>, u_steps: usize, u_max: Option<T>) -> Self {
    assert!(
      alpha > T::zero() && alpha < T::one(),
      "alpha must be in (0,1)"
    );
    assert!(c > T::zero(), "c must be positive");
    assert!(u_steps >= 2, "u_steps must be >= 2");
    Self {
      alpha,
      c,
      n,
      t,
      u_steps,
      u_max,
    }
  }

  fn simulate_direct_path(&self, u_max: f64, rng: &mut impl Rng) -> (Vec<f64>, Vec<f64>) {
    let m = self.u_steps;
    let du = u_max / (m - 1) as f64;
    let alpha = self.alpha.to_f64().unwrap();
    let c = self.c.to_f64().unwrap();
    let scale = (c * du).powf(1.0 / alpha);
    let mut u = vec![0.0; m];
    let mut d = vec![0.0; m];
    for i in 1..m {
      u[i] = i as f64 * du;
      d[i] = d[i - 1] + scale * sample_positive_stable(alpha, rng);
    }
    (u, d)
  }
}

impl<T: FloatExt> ProcessExt<T> for InverseAlphaStableSubordinator<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut out = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return out;
    }
    if self.n == 1 {
      out[0] = T::zero();
      return out;
    }

    let t_max = self.t.unwrap_or(T::one()).to_f64().unwrap();
    let mut u_max = self
      .u_max
      .unwrap_or(T::from_f64_fast(t_max))
      .to_f64()
      .unwrap();
    if u_max <= 0.0 {
      u_max = t_max.max(1.0);
    }

    let mut rng = rand::rng();
    let mut u = Vec::new();
    let mut d = Vec::new();
    let mut reached = false;

    for _ in 0..10 {
      let (u_try, d_try) = self.simulate_direct_path(u_max, &mut rng);
      if *d_try.last().unwrap_or(&0.0) >= t_max {
        u = u_try;
        d = d_try;
        reached = true;
        break;
      }
      u_max *= 2.0;
      u = u_try;
      d = d_try;
    }

    let dt = t_max / (self.n - 1) as f64;
    let mut j = 1usize;
    for i in 0..self.n {
      let t_i = i as f64 * dt;
      while j < d.len() && d[j] < t_i {
        j += 1;
      }
      let e_i = if j >= d.len() {
        *u.last().unwrap_or(&u_max)
      } else if d[j] <= d[j - 1] {
        u[j]
      } else {
        let w = (t_i - d[j - 1]) / (d[j] - d[j - 1]);
        u[j - 1] + w * (u[j] - u[j - 1])
      };
      out[i] = T::from_f64_fast(e_i);
    }

    if !reached {
      for x in &mut out {
        if !x.is_finite() {
          *x = T::from_f64_fast(u_max);
        }
      }
    }

    out
  }
}

py_process_1d!(PyInverseAlphaStableSubordinator, InverseAlphaStableSubordinator,
  sig: (alpha, c, n, t=None, u_steps=2048, u_max=None, dtype=None),
  params: (alpha: f64, c: f64, n: usize, t: Option<f64>, u_steps: usize, u_max: Option<f64>)
);
