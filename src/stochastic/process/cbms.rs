use ndarray::Array1;

use crate::stochastic::noise::cgns::CGNS;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct CBMS<T: FloatExt> {
  pub rho: T,
  pub n: usize,
  pub t: Option<T>,
  cgns: CGNS<T>,
}

impl<T: FloatExt> CBMS<T> {
  pub fn new(rho: T, n: usize, t: Option<T>) -> Self {
    assert!(
      (-T::one()..=T::one()).contains(&rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    Self {
      rho,
      n,
      t,
      cgns: CGNS::new(rho, n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for CBMS<T> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let [cgn1, cgn2] = &self.cgns.sample();

    let mut bm1 = Array1::<T>::zeros(self.n);
    let mut bm2 = Array1::<T>::zeros(self.n);

    for i in 1..self.n {
      bm1[i] = bm1[i - 1] + cgn1[i - 1];
      bm2[i] = bm2[i - 1] + cgn2[i - 1];
    }

    [bm1, bm2]
  }
}
