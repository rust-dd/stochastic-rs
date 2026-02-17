use ndarray::Array1;
use ndarray_rand::RandomExt;

use crate::distributions::inverse_gauss::SimdInverseGauss;
use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct NIG<T: FloatExt> {
  pub theta: T,
  pub sigma: T,
  pub kappa: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: FloatExt> NIG<T> {
  pub fn new(theta: T, sigma: T, kappa: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      theta,
      sigma,
      kappa,
      n,
      x0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for NIG<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let gn = &self.gn.sample();
    let dt = self.gn.dt();
    let scale = dt.powf(T::from_usize_(2)) / self.kappa;
    let mean = dt / scale;
    let ig_dist = SimdInverseGauss::new(mean, scale);
    let ig = Array1::random(self.n - 1, &ig_dist);
    let mut nig = Array1::zeros(self.n);
    nig[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      nig[i] = nig[i - 1] + self.theta * ig[i - 1] + self.sigma * ig[i - 1].sqrt() * gn[i - 1]
    }

    nig
  }
}

py_process_1d!(PyNIG, NIG,
  sig: (theta, sigma, kappa, n, x0=None, t=None, dtype=None),
  params: (theta: f64, sigma: f64, kappa: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
