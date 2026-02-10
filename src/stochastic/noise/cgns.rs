use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

#[derive(Copy, Clone)]
pub struct CGNS<T: Float> {
  pub rho: T,
  pub n: usize,
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: Float> CGNS<T> {
  pub fn new(rho: T, n: usize, t: Option<T>) -> Self {
    assert!(
      (-1.0..=1.0).contains(&rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    Self {
      rho,
      n,
      t,
      gn: Gn::new(n, t),
    }
  }
}

impl<T: Float> Process<T> for CGNS<T> {
  type Output = [Array1<T>; 2];
  type Noise = Gn<T>;

  fn sample(&self) -> Self::Output {
    self.euler_maruyama(|gn| gn.sample())
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    self.euler_maruyama(|gn| gn.sample_simd())
  }

  fn euler_maruyama(
    &self,
    noise_fn: impl Fn(&Self::Noise) -> <Self::Noise as Process<T>>::Output,
  ) -> Self::Output {
    let gn1 = noise_fn(&self.gn);
    let z = noise_fn(&self.gn);
    let c = (T::one() - self.rho.powi(2)).sqrt();
    let mut gn2 = Array1::zeros(self.n);

    for i in 0..self.n {
      gn2[i] = self.rho * gn1[i] + c * z[i];
    }

    [gn1, gn2]
  }
}

impl<T: Float> CGNS<T> {
  pub fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize(self.n)
  }
}
