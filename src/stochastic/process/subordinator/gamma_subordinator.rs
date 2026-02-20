use ndarray::Array1;
use ndarray_rand::RandomExt;

use crate::distributions::gamma::SimdGamma;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Gamma subordinator where `G_t ~ Gamma(nu * t, rate)`.
pub struct GammaSubordinator<T: FloatExt> {
  /// Shape intensity `nu`.
  pub nu: T,
  /// Rate parameter (>0).
  pub rate: T,
  /// Number of grid points.
  pub n: usize,
  /// Initial level.
  pub x0: Option<T>,
  /// Horizon.
  pub t: Option<T>,
}

impl<T: FloatExt> GammaSubordinator<T> {
  pub fn new(nu: T, rate: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    assert!(nu > T::zero(), "nu must be positive");
    assert!(rate > T::zero(), "rate must be positive");
    Self { nu, rate, n, x0, t }
  }
}

impl<T: FloatExt> ProcessExt<T> for GammaSubordinator<T> {
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
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1);
    let shape = self.nu * dt;
    let scale = T::one() / self.rate;
    let gamma = SimdGamma::new(shape, scale);
    let inc = Array1::random(self.n - 1, &gamma);
    for i in 1..self.n {
      out[i] = out[i - 1] + inc[i - 1];
    }
    out
  }
}

py_process_1d!(PyGammaSubordinator, GammaSubordinator,
  sig: (nu, rate, n, x0=None, t=None, dtype=None),
  params: (nu: f64, rate: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
