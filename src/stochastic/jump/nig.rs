use ndarray::Array1;
use ndarray_rand::RandomExt;

use crate::distributions::inverse_gauss::SimdInverseGauss;
use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct NIG<T: Float> {
  pub theta: T,
  pub sigma: T,
  pub kappa: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  ig: SimdInverseGauss<T>,
  gn: Gn<T>,
}

unsafe impl<T: Float> Send for NIG<T> {}
unsafe impl<T: Float> Sync for NIG<T> {}

impl<T: Float> NIG<T> {
  pub fn new(theta: T, sigma: T, kappa: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    let gn = Gn::new(n - 1, t);
    let dt = gn.dt();
    let scale = dt.powf(T::from_usize(2).unwrap()) / kappa;
    let mean = dt / scale;
    let ig = SimdInverseGauss::new(mean, scale);

    Self {
      theta,
      sigma,
      kappa,
      n,
      x0,
      t,
      ig,
      gn,
    }
  }
}

impl<T: Float> Process<T> for NIG<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let gn = &self.gn.sample();
    let ig = Array1::random(self.n - 1, &self.ig);
    let mut nig = Array1::zeros(self.n);
    nig[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      nig[i] = nig[i - 1] + self.theta * ig[i - 1] + self.sigma * ig[i - 1].sqrt() * gn[i - 1]
    }

    nig
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::plot_1d;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn nig_length_equals_n() {
    let nig = NIG::new(2.25, 2.5, 1.0, N, Some(X0), Some(100.0));
    assert_eq!(nig.sample().len(), N);
  }

  #[test]
  fn nig_starts_with_x0() {
    let nig = NIG::new(2.25, 2.5, 1.0, N, Some(X0), Some(100.0));
    assert_eq!(nig.sample()[0], X0);
  }

  #[test]
  fn nig_plot() {
    let nig = NIG::new(2.25, 2.5, 1.0, N, Some(X0), Some(100.0));
    plot_1d!(nig.sample(), "Normal Inverse Gaussian (NIG)");
  }
}
