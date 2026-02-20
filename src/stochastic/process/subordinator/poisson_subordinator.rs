use ndarray::Array1;
use rand_distr::Distribution;

use crate::distributions::poisson::SimdPoisson;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Poisson subordinator with unit jumps:
/// `N_t` with independent increments `Poisson(lambda * dt)`.
pub struct PoissonSubordinator<T: FloatExt> {
  /// Intensity parameter.
  pub lambda: T,
  /// Number of grid points.
  pub n: usize,
  /// Initial level.
  pub x0: Option<T>,
  /// Horizon.
  pub t: Option<T>,
}

impl<T: FloatExt> PoissonSubordinator<T> {
  pub fn new(lambda: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    assert!(lambda > T::zero(), "lambda must be positive");
    Self { lambda, n, x0, t }
  }
}

impl<T: FloatExt> ProcessExt<T> for PoissonSubordinator<T> {
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
    let t_max = self.t.unwrap_or(T::one());
    let dt = t_max / T::from_usize_(self.n - 1);
    let lambda_dt = (self.lambda * dt).to_f64().unwrap();
    let poisson = SimdPoisson::<u32>::new(lambda_dt);
    let mut rng = rand::rng();
    for i in 1..self.n {
      let k = poisson.sample(&mut rng) as usize;
      out[i] = out[i - 1] + T::from_usize_(k);
    }
    out
  }
}

py_process_1d!(PyPoissonSubordinator, PoissonSubordinator,
  sig: (lambda_, n, x0=None, t=None, dtype=None),
  params: (lambda_: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
