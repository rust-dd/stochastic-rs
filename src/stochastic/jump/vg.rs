use ndarray::Array1;
use ndarray_rand::RandomExt;

#[cfg(feature = "simd")]
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

    #[cfg(not(feature = "simd"))]
    let gamma = Gamma::new(shape, scale).unwrap();
    #[cfg(feature = "simd")]
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

#[cfg(test)]
mod tests {
  use super::*;
  use crate::plot_1d;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn vg_length_equals_n() {
    let vg = VG::new(2.25, 2.5, 1.0, N, Some(X0), None);
    assert_eq!(vg.sample().len(), N);
  }

  #[test]
  fn vg_starts_with_x0() {
    let vg = VG::new(2.25, 2.5, 1.0, N, Some(X0), None);
    assert_eq!(vg.sample()[0], X0);
  }

  #[test]
  fn vg_plot() {
    let vg = VG::new(2.25, 2.5, 1.0, N, Some(X0), None);
    plot_1d!(vg.sample(), "Variace Gamma (VG)");
  }
}
