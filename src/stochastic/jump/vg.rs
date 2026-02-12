use ndarray::Array1;
use ndarray_rand::RandomExt;

use crate::distributions::gamma::SimdGamma;
use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct VG<T: Float> {
  pub mu: T,
  pub sigma: T,
  pub nu: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  gamma: SimdGamma<T>,
  gn: Gn<T>,
}

unsafe impl<T: Float> Send for VG<T> {}
unsafe impl<T: Float> Sync for VG<T> {}

impl<T: Float> VG<T> {
  pub fn new(mu: T, sigma: T, nu: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    let gn = Gn::new(n - 1, t);
    let dt = gn.dt();
    let shape = dt / nu;
    let scale = nu;
    let gamma = SimdGamma::new(shape, scale);

    Self {
      mu,
      sigma,
      nu,
      n,
      x0,
      t,
      gamma: gamma.into(),
      gn,
    }
  }
}

impl<T: Float> Process<T> for VG<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut vg = Array1::<T>::zeros(self.n);
    vg[0] = self.x0.unwrap_or(T::zero());

    let gn = &self.gn.sample();
    let gammas = Array1::random(self.n - 1, &self.gamma);

    for i in 1..self.n {
      vg[i] = vg[i - 1] + self.mu * gammas[i - 1] + self.sigma * gammas[i - 1].sqrt() * gn[i - 1];
    }

    vg
  }
}
