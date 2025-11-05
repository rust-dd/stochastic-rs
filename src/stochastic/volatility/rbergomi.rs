use impl_new_derive::ImplNew;
use ndarray::{s, Array1};

use crate::stochastic::{noise::cgns::CGNS, Sampling2DExt};

#[derive(ImplNew)]
pub struct RoughBergomi<T> {
  pub hurst: T,
  pub nu: T,
  pub v0: Option<T>,
  pub s0: Option<T>,
  pub r: T,
  pub rho: T,
  pub n: usize,
  pub t: Option<T>,
  pub m: Option<usize>,
  pub cgns: CGNS<T>,
}

impl Sampling2DExt<f64> for RoughBergomi<f64> {
  fn sample(&self) -> [Array1<f64>; 2] {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let [cgn1, z] = self.cgns.sample();

    let mut s = Array1::<f64>::zeros(self.n);
    let mut v2 = Array1::<f64>::zeros(self.n);
    s[0] = self.s0.unwrap_or(100.0);
    v2[0] = self.v0.unwrap_or(1.0).powi(2);

    for i in 1..=self.n {
      s[i] = s[i - 1] + self.r * s[i - 1] * dt + v2[i - 1].sqrt() * s[i - 1] * cgn1[i - 1];

      let sum_z = z.slice(s![..i]).sum();
      let t = i as f64 * dt;
      v2[i] = self.v0.unwrap_or(1.0).powi(2)
        * (self.nu * (2.0 * self.hurst).sqrt() * t.powf(self.hurst - 0.5) * sum_z
          - 0.5 * self.nu.powi(2) * t.powf(2.0 * self.hurst))
        .exp();
    }

    [s, v2]
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
