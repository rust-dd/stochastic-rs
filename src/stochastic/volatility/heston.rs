#[cfg(feature = "malliavin")]
use std::sync::Mutex;

use impl_new_derive::ImplNew;
use ndarray::Array1;

use crate::stochastic::{noise::cgns::CGNS, Sampling2DExt};

use super::HestonPow;

#[derive(ImplNew)]

pub struct Heston<T> {
  /// Initial stock price
  pub s0: Option<T>,
  /// Initial volatility
  pub v0: Option<T>,
  /// Mean reversion rate
  pub kappa: T,
  /// Long-run average volatility
  pub theta: T,
  /// Volatility of volatility
  pub sigma: T,
  /// Correlation between the stock price and its volatility
  pub rho: T,
  /// Drift of the stock price
  pub mu: T,
  /// Number of time steps
  pub n: usize,
  /// Time to maturity
  pub t: Option<T>,
  /// Power of the variance
  /// If 0.5 then it is the original Heston model
  /// If 1.5 then it is the 3/2 model
  pub pow: HestonPow,
  /// Use the symmetric method for the variance to avoid negative values
  pub use_sym: Option<bool>,
  /// Number of paths for multithreading
  pub m: Option<usize>,
  /// Noise generator
  pub cgns: CGNS<T>,
  /// Calculate the Malliavin derivative
  #[cfg(feature = "malliavin")]
  pub calculate_malliavin: Option<bool>,
  /// Malliavin derivative of the volatility
  #[cfg(feature = "malliavin")]
  malliavin_of_vol: Mutex<Option<Array1<T>>>,
  /// Malliavin derivative of the price
  #[cfg(feature = "malliavin")]
  malliavin_of_price: Mutex<Option<Array1<T>>>,
}

impl Sampling2DExt<f64> for Heston<f64> {
  fn sample(&self) -> [Array1<f64>; 2] {
    let [cgn1, cgn2] = self.cgns.sample();
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;

    let mut s = Array1::<f64>::zeros(self.n);
    let mut v = Array1::<f64>::zeros(self.n);

    s[0] = self.s0.unwrap_or(0.0);
    v[0] = self.v0.unwrap_or(0.0);

    for i in 1..self.n {
      s[i] = s[i - 1] + self.mu * s[i - 1] * dt + s[i - 1] * v[i - 1].sqrt() * cgn1[i - 1];

      let dv = self.kappa * (self.theta - v[i - 1]) * dt
        + self.sigma
          * v[i - 1].powf(match self.pow {
            HestonPow::Sqrt => 0.5,
            HestonPow::ThreeHalves => 1.5,
          })
          * cgn2[i - 1];

      v[i] = match self.use_sym.unwrap_or(false) {
        true => (v[i - 1] + dv).abs(),
        false => (v[i - 1] + dv).max(0.0),
      }
    }

    #[cfg(feature = "malliavin")]
    if self.calculate_malliavin.is_some() && self.calculate_malliavin.unwrap() {
      let mut det_term = Array1::zeros(self.n);
      let mut malliavin = Array1::zeros(self.n);

      for i in 0..self.n {
        match self.pow {
          HestonPow::Sqrt => {
            det_term[i] = ((-(self.kappa * self.theta / 2.0 - self.sigma.powi(2) / 8.0)
              * (1.0 / v.last().unwrap())
              - self.kappa / 2.0)
              * ((self.n - i) as f64 * dt))
              .exp();
            malliavin[i] = (self.sigma * v.last().unwrap().sqrt() / 2.0) * det_term[i];
          }
          HestonPow::ThreeHalves => {
            det_term[i] = ((-(self.kappa * self.theta / 2.0 + 3.0 * self.sigma.powi(2) / 8.0)
              * v.last().unwrap()
              - (self.kappa * self.theta) / 2.0)
              * ((self.n - i) as f64 * dt))
              .exp();
            malliavin[i] = (self.sigma * v.last().unwrap().powf(1.5) / 2.0) * det_term[i];
          }
        };
      }

      let _ = std::mem::replace(&mut *self.malliavin_of_vol.lock().unwrap(), Some(malliavin));
    }

    [s, v]
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }

  /// Malliavin derivative of the volatility
  ///
  /// The Malliavin derivative of the Heston model is given by
  /// D_r v_t = \sigma v_t^{1/2} / 2 * exp(-(\kappa \theta / 2 - \sigma^2 / 8) / v_t * dt)
  ///
  /// The Malliavin derivative of the 3/2 Heston model is given by
  /// D_r v_t = \sigma v_t^{3/2} / 2 * exp(-(\kappa \theta / 2 + 3 \sigma^2 / 8) * v_t * dt)
  #[cfg(feature = "malliavin")]
  fn malliavin(&self) -> [Array1<f64>; 2] {
    [
      Array1::zeros(self.n),
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

#[cfg(feature = "f32")]
impl Sampling2DExt<f32> for Heston<f32> {
  fn sample(&self) -> [Array1<f32>; 2] {
    let [cgn1, cgn2] = self.cgns.sample();
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;

    let mut s = Array1::<f32>::zeros(self.n);
    let mut v = Array1::<f32>::zeros(self.n);

    s[0] = self.s0.unwrap_or(0.0);
    v[0] = self.v0.unwrap_or(0.0);

    for i in 1..self.n {
      s[i] = s[i - 1] + self.mu * s[i - 1] * dt + s[i - 1] * v[i - 1].sqrt() * cgn1[i - 1];

      let dv = self.kappa * (self.theta - v[i - 1]) * dt
        + self.sigma
          * v[i - 1].powf(match self.pow {
            HestonPow::Sqrt => 0.5,
            HestonPow::ThreeHalves => 1.5,
          })
          * cgn2[i - 1];

      v[i] = match self.use_sym.unwrap_or(false) {
        true => (v[i - 1] + dv).abs(),
        false => (v[i - 1] + dv).max(0.0),
      }
    }

    [s, v]
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

#[cfg(test)]
mod tests {
  #[cfg(feature = "malliavin")]
  use crate::{
    plot_2d,
    stochastic::{N, S0, X0},
  };

  #[cfg(feature = "malliavin")]
  use super::*;

  #[test]
  #[cfg(feature = "malliavin")]
  fn heston_malliavin() {
    let heston = Heston::new(
      Some(S0),
      Some(X0),
      0.5,
      1.0,
      1.0,
      1.0,
      1.0,
      N,
      Some(1.0),
      HestonPow::Sqrt,
      None,
      None,
      CGNS::new(0.7, N, None, None),
      Some(true),
    );
    let process = heston.sample();
    let malliavin = heston.malliavin();
    plot_2d!(
      process[1],
      "Heston volatility process",
      malliavin[1],
      "Malliavin derivative of the Heston volatility process"
    );
  }
}
