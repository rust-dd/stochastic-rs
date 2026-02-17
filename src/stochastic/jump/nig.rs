use ndarray::Array1;
use ndarray_rand::RandomExt;

use crate::distributions::inverse_gauss::SimdInverseGauss;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct NIG<T: FloatExt> {
  pub theta: T,
  pub sigma: T,
  pub kappa: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
}

impl<T: FloatExt> NIG<T> {
  pub fn new(theta: T, sigma: T, kappa: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    assert!(kappa > T::zero(), "kappa must be positive");
    Self {
      theta,
      sigma,
      kappa,
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

impl<T: FloatExt> ProcessExt<T> for NIG<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut nig = Array1::zeros(self.n);
    if self.n <= 1 {
      return nig;
    }
    nig[0] = self.x0.unwrap_or(T::zero());

    let dt = self.dt();
    // For NIG: G_dt ~ IG(mean=dt, shape=dt^2/kappa).
    let shape = dt * dt / self.kappa;
    let ig_dist = SimdInverseGauss::new(dt, shape);
    let ig = Array1::random(self.n - 1, &ig_dist);
    let mut z = Array1::<T>::zeros(self.n - 1);
    let z_slice = z.as_slice_mut().expect("NIG normals must be contiguous");
    T::fill_standard_normal_slice(z_slice);

    for i in 1..self.n {
      nig[i] = nig[i - 1] + self.theta * ig[i - 1] + self.sigma * ig[i - 1].sqrt() * z[i - 1]
    }

    nig
  }
}

py_process_1d!(PyNIG, NIG,
  sig: (theta, sigma, kappa, n, x0=None, t=None, dtype=None),
  params: (theta: f64, sigma: f64, kappa: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
