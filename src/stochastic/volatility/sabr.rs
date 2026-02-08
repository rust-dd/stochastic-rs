use std::sync::Mutex;

use impl_new_derive::ImplNew;
use ndarray::Array1;

use crate::stochastic::noise::cgns::CGNS;
use crate::stochastic::Sampling2DExt;

#[derive(ImplNew)]

pub struct SABR<T> {
  pub alpha: T,
  pub beta: T,
  pub rho: T,
  pub n: usize,
  pub f0: Option<T>,
  pub v0: Option<T>,
  pub t: Option<T>,
  pub m: Option<usize>,
  pub cgns: CGNS<T>,
  pub calculate_malliavin: Option<bool>,
  malliavin_of_vol: Mutex<Option<Array1<T>>>,
  malliavin_of_price: Mutex<Option<Array1<T>>>,
}

impl Sampling2DExt<f64> for SABR<f64> {
  fn sample(&self) -> [Array1<f64>; 2] {
    let [cgn1, cgn2] = self.cgns.sample();

    let mut f = Array1::<f64>::zeros(self.n);
    let mut v = Array1::<f64>::zeros(self.n);

    f[0] = self.f0.unwrap_or(0.0);
    v[0] = self.v0.unwrap_or(0.0);

    for i in 1..self.n {
      f[i] = f[i - 1] + v[i - 1] * f[i - 1].powf(self.beta) * cgn1[i - 1];
      v[i] = v[i - 1] + self.alpha * v[i - 1] * cgn2[i - 1];
    }

    if self.calculate_malliavin.is_some() && self.calculate_malliavin.unwrap() {
      // Only volatility Malliavin derivative is supported
      let mut malliavin_of_vol = Array1::<f64>::zeros(self.n);

      for i in 0..self.n {
        malliavin_of_vol[i] = self.alpha * v.last().unwrap();
      }

      let _ = std::mem::replace(
        &mut *self.malliavin_of_vol.lock().unwrap(),
        Some(malliavin_of_vol),
      );
    }

    [f, v]
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }

  /// Calculate the Malliavin derivative of the SABR model
  ///
  /// The Malliavin derivative of the volaility process in the SABR model is given by:
  /// D_r \sigma_t = \alpha \sigma_t 1_{[0, T]}(r)
  fn malliavin(&self) -> [Array1<f64>; 2] {
    [
      Array1::zeros(self.n + 1),
      self
        .malliavin_of_vol
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .clone(),
    ]
  }
}

impl Sampling2DExt<f32> for SABR<f32> {
  fn sample(&self) -> [Array1<f32>; 2] {
    let [cgn1, cgn2] = self.cgns.sample();

    let mut f = Array1::<f32>::zeros(self.n);
    let mut v = Array1::<f32>::zeros(self.n);

    f[0] = self.f0.unwrap_or(0.0);
    v[0] = self.v0.unwrap_or(0.0);

    for i in 1..self.n {
      f[i] = f[i - 1] + v[i - 1] * f[i - 1].powf(self.beta) * cgn1[i - 1];
      v[i] = v[i - 1] + self.alpha * v[i - 1] * cgn2[i - 1];
    }

    [f, v]
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(test)]

mod tests {
  use super::*;
  use crate::plot_2d;
  use crate::stochastic::N;

  #[test]
  fn sabr_malliavin() {
    let sabr = SABR::new(
      0.5,
      0.5,
      0.5,
      N,
      Some(1.0),
      Some(1.0),
      Some(1.0),
      None,
      CGNS::new(0.7, N, None, None),
      Some(true),
    );
    let process = sabr.sample();
    let malliavin = sabr.malliavin();
    plot_2d!(
      process[1],
      "SABR volatility process",
      malliavin[1],
      "Malliavin derivative of the SABR volatility process"
    );
  }
}
