use ndarray::Array1;
use ndarray_rand::RandomExt;

use crate::distributions::gamma::SimdGamma;
use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct VG<T: FloatExt> {
  pub mu: T,
  pub sigma: T,
  pub nu: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: FloatExt> VG<T> {
  pub fn new(mu: T, sigma: T, nu: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      mu,
      sigma,
      nu,
      n,
      x0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for VG<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut vg = Array1::<T>::zeros(self.n);
    vg[0] = self.x0.unwrap_or(T::zero());

    let gn = &self.gn.sample();
    let dt = self.gn.dt();
    let gamma = SimdGamma::new(dt / self.nu, self.nu);
    let gammas = Array1::random(self.n - 1, &gamma);

    for i in 1..self.n {
      vg[i] = vg[i - 1] + self.mu * gammas[i - 1] + self.sigma * gammas[i - 1].sqrt() * gn[i - 1];
    }

    vg
  }
}

py_process_1d!(PyVG, VG,
  sig: (mu, sigma, nu, n, x0=None, t=None, dtype=None),
  params: (mu: f64, sigma: f64, nu: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
