use impl_new_derive::ImplNew;
use ndarray::{Array1, Array2};

use crate::stochastic::{noise::cfgns::CFGNS, Sampling2DExt};

#[derive(ImplNew)]
pub struct CFBMS<T> {
  pub rho: T,
  pub n: usize,
  pub t: Option<T>,
  pub m: Option<usize>,
  pub cfgns: CFGNS<T>,
}

impl Sampling2DExt<f64> for CFBMS<f64> {
  fn sample(&self) -> [Array1<f64>; 2] {
    assert!(
      (-1.0..=1.0).contains(&self.rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    let mut fbms = Array2::<f64>::zeros((2, self.n));
    let [fgn1, fgn2] = self.cfgns.sample();

    for i in 1..self.n {
      fbms[[0, i]] = fbms[[0, i - 1]] + fgn1[i - 1];
      fbms[[1, i]] =
        fbms[[1, i - 1]] + self.rho * fgn1[i - 1] + (1.0 - self.rho.powi(2)).sqrt() * fgn2[i - 1];
    }

    [fbms.row(0).to_owned(), fbms.row(1).to_owned()]
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
