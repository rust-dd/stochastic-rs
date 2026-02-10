use ndarray::Array1;

use super::fgn::FGN;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct CFGNS<T: Float> {
  pub hurst: T,
  pub rho: T,
  pub n: usize,
  pub t: Option<T>,
  fgn: FGN<T>,
}

impl<T: Float> CFGNS<T> {
  pub fn new(hurst: T, rho: T, n: usize, t: Option<T>) -> Self {
    assert!(
      (0.0..=1.0).contains(&hurst),
      "Hurst parameter must be in (0, 1)"
    );
    assert!(
      (-1.0..=1.0).contains(&rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    Self {
      hurst,
      rho,
      n,
      t,
      fgn: FGN::new(hurst, n, t),
    }
  }
}

impl<T: Float> Process<T> for CFGNS<T> {
  type Output = [Array1<T>; 2];
  type Noise = FGN<T>;

  fn sample(&self) -> Self::Output {
    self.euler_maruyama(|fgn| fgn.sample())
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    self.euler_maruyama(|fgn| fgn.sample_simd())
  }

  fn euler_maruyama(
    &self,
    noise_fn: impl Fn(&Self::Noise) -> <Self::Noise as Process<T>>::Output,
  ) -> Self::Output {
    let fgn1 = noise_fn(&self.fgn);
    let z = noise_fn(&self.gn);
    let c = (T::one() - self.rho.powi(2)).sqrt();
    let mut fgn2 = Array1::new(self.n);

    for i in 0..self.n {
      fgn2[i] = self.rho * fgn1[i] + c * z[i];
    }

    [fgn1, fgn2]
  }
}
