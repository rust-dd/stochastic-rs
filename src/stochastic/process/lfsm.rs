use ndarray::Array1;
use ndarray_rand::RandomExt;

use crate::distributions::alpha_stable::SimdAlphaStable;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Linear fractional stable motion (LFSM), also commonly referred to as
/// Levy fractional stable motion in discretized form.
///
/// The implementation uses a one-sided moving-average discretization with
/// alpha-stable innovations:
///
/// `X_i = X_{i-1} + sum_{k=0}^{i-1} w_k * xi_{i-1-k}`,
/// where `w_k = dt^d * ((k+1)^d - k^d)` and `d = H - 1/alpha`.
pub struct LFSM<T: FloatExt> {
  pub alpha: T,
  pub beta: T,
  pub hurst: T,
  pub scale: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
}

impl<T: FloatExt> LFSM<T> {
  pub fn new(alpha: T, beta: T, hurst: T, scale: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    assert!(alpha > T::zero() && alpha <= T::from(2.0).unwrap());
    assert!((-T::one()..=T::one()).contains(&beta));
    assert!(scale > T::zero());
    assert!(
      hurst > T::one() / alpha && hurst < T::one(),
      "LFSM requires 1/alpha < hurst < 1 for this discretization"
    );
    Self {
      alpha,
      beta,
      hurst,
      scale,
      n,
      x0,
      t,
    }
  }

  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }
}

impl<T: FloatExt> ProcessExt<T> for LFSM<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut x = Array1::<T>::zeros(self.n);
    if self.n <= 1 {
      return x;
    }
    x[0] = self.x0.unwrap_or(T::zero());

    let dt = self.dt();
    let d = self.hurst - T::one() / self.alpha;
    let kernel_scale = dt.powf(d);
    let innovation_scale = self.scale * dt.powf(T::one() / self.alpha);

    let stable = SimdAlphaStable::new(self.alpha, self.beta, innovation_scale, T::zero());
    let innovations = Array1::random(self.n - 1, &stable);

    let mut weights = Array1::<T>::zeros(self.n - 1);
    for k in 0..(self.n - 1) {
      let kf = T::from_usize_(k);
      weights[k] = kernel_scale * ((kf + T::one()).powf(d) - kf.powf(d));
    }

    for i in 1..self.n {
      let mut inc = T::zero();
      for k in 0..i {
        inc += weights[k] * innovations[i - 1 - k];
      }
      x[i] = x[i - 1] + inc;
    }

    x
  }
}

py_process_1d!(PyLFSM, LFSM,
  sig: (alpha, beta, hurst, scale, n, x0=None, t=None, dtype=None),
  params: (alpha: f64, beta: f64, hurst: f64, scale: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn lfsm_path_is_finite() {
    let p = LFSM::new(1.7_f64, 0.0, 0.8, 1.0, 256, Some(0.0), Some(1.0));
    let x = p.sample();
    assert_eq!(x.len(), 256);
    assert!(x.iter().all(|v| v.is_finite()));
  }
}
