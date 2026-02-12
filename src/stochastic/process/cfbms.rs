use ndarray::Array1;

use crate::stochastic::noise::cfgns::CFGNS;
use crate::stochastic::Float;
use crate::stochastic::ProcessExt;

pub struct CFBMS<T: Float> {
  pub hurst: T,
  pub rho: T,
  pub n: usize,
  pub t: Option<T>,
  pub m: Option<usize>,
  cfgns: CFGNS<T>,
}

impl<T: Float> CFBMS<T> {
  pub fn new(hurst: T, rho: T, n: usize, t: Option<T>, m: Option<usize>) -> Self {
    assert!(
      (T::zero()..=T::one()).contains(&hurst),
      "Hurst parameter must be in (0, 1)"
    );
    assert!(
      (-T::one()..=T::one()).contains(&rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    Self {
      hurst,
      rho,
      n,
      t,
      m,
      cfgns: CFGNS::new(hurst, rho, n - 1, t),
    }
  }
}

impl<T: Float> ProcessExt<T> for CFBMS<T> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let [fgn1, fgn2] = &self.cfgns.sample();

    let mut fbm1 = Array1::<T>::zeros(self.n);
    let mut fbm2 = Array1::<T>::zeros(self.n);

    for i in 1..self.n {
      fbm1[i] = fbm1[i - 1] + fgn1[i - 1];
      fbm2[i] = fbm2[i - 1] + fgn2[i - 1];
    }

    [fbm1, fbm2]
  }
}
