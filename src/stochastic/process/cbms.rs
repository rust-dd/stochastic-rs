use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray::Array2;

use crate::stochastic::noise::cgns::CGNS;
use crate::stochastic::Sampling2DExt;

#[derive(ImplNew)]
pub struct CBMS<T> {
  pub rho: T,
  pub n: usize,
  pub t: Option<T>,
  pub m: Option<usize>,
  pub cgns: CGNS<T>,
}

#[cfg(feature = "f64")]
impl Sampling2DExt<f64> for CBMS<f64> {
  fn sample(&self) -> [Array1<f64>; 2] {
    assert!(
      (-1.0..=1.0).contains(&self.rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    let mut bms = Array2::<f64>::zeros((2, self.n));
    let [cgn1, cgn2] = self.cgns.sample();

    for i in 1..self.n {
      bms[[0, i]] = bms[[0, i - 1]] + cgn1[i - 1];
      bms[[1, i]] =
        bms[[1, i - 1]] + self.rho * cgn1[i - 1] + (1.0 - self.rho.powi(2)).sqrt() * cgn2[i - 1];
    }

    [bms.row(0).into_owned(), bms.row(1).into_owned()]
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(feature = "f32")]
impl Sampling2DExt<f32> for CBMS<f32> {
  fn sample(&self) -> [Array1<f32>; 2] {
    assert!(
      (-1.0..=1.0).contains(&self.rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    let mut bms = Array2::<f32>::zeros((2, self.n));
    let [cgn1, cgn2] = self.cgns.sample();

    for i in 1..self.n {
      bms[[0, i]] = bms[[0, i - 1]] + cgn1[i - 1];
      bms[[1, i]] =
        bms[[1, i - 1]] + self.rho * cgn1[i - 1] + (1.0 - self.rho.powi(2)).sqrt() * cgn2[i - 1];
    }

    [bms.row(0).into_owned(), bms.row(1).into_owned()]
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}
