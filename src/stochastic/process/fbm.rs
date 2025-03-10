#[cfg(feature = "malliavin")]
use std::sync::Mutex;

use impl_new_derive::ImplNew;
use ndarray::{s, Array1};
#[cfg(feature = "malliavin")]
use statrs::function::gamma;

use crate::stochastic::{noise::fgn::FGN, Sampling};

#[derive(ImplNew)]
pub struct FBM {
  pub hurst: f64,
  pub n: usize,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub fgn: FGN,
  #[cfg(feature = "malliavin")]
  pub calculate_malliavin: Option<bool>,
  #[cfg(feature = "malliavin")]
  malliavin: Mutex<Option<Array1<f64>>>,
}

impl Sampling<f64> for FBM {
  fn sample(&self) -> Array1<f64> {
    let fgn = self.fgn.sample();
    let mut fbm = Array1::<f64>::zeros(self.n);
    fbm.slice_mut(s![1..]).assign(&fgn);

    for i in 1..self.n {
      fbm[i] += fbm[i - 1];
    }

    #[cfg(feature = "malliavin")]
    if self.calculate_malliavin.is_some() && self.calculate_malliavin.unwrap() {
      let mut malliavin = Array1::zeros(self.n);
      let dt = self.t.unwrap_or(1.0) / (self.n) as f64;
      for i in 0..self.n {
        malliavin[i] =
          1.0 / (gamma::gamma(self.hurst + 0.5)) * (i as f64 * dt).powf(self.hurst - 0.5);
      }

      let _ = std::mem::replace(&mut *self.malliavin.lock().unwrap(), Some(malliavin));
    }
    fbm.slice(s![..self.n()]).to_owned()
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }

  /// Calculate the Malliavin derivative
  ///
  /// The Malliavin derivative of the fractional Brownian motion is given by:
  /// D_s B^H_t = 1 / Γ(H + 1/2) (t - s)^{H - 1/2}
  ///
  /// where B^H_t is the fractional Brownian motion with Hurst parameter H in Mandelbrot-Van Ness representation as
  /// B^H_t = 1 / Γ(H + 1/2) ∫_0^t (t - s)^{H - 1/2} dW_s
  /// which is a truncated Wiener integral.
  #[cfg(feature = "malliavin")]
  fn malliavin(&self) -> Array1<f64> {
    self.malliavin.lock().unwrap().clone().unwrap()
  }
}

#[cfg(test)]
mod tests {
  #[cfg(feature = "malliavin")]
  use crate::plot_2d;
  use crate::{plot_1d, stochastic::N};

  use super::*;

  #[test]
  fn fbm_length_equals_n() {
    let fbm = FBM::new(
      0.7,
      N,
      Some(1.0),
      None,
      FGN::new(0.7, N - 1, Some(1.0), None),
      #[cfg(feature = "malliavin")]
      None,
    );

    assert_eq!(fbm.sample().len(), N);
  }

  #[test]
  fn fbm_starts_with_x0() {
    let fbm = FBM::new(
      0.7,
      N,
      Some(1.0),
      None,
      FGN::new(0.7, N - 1, Some(1.0), None),
      #[cfg(feature = "malliavin")]
      None,
    );

    assert_eq!(fbm.sample()[0], 0.0);
  }

  #[test]
  fn fbm_plot() {
    let fbm = FBM::new(
      0.7,
      N,
      Some(1.0),
      None,
      FGN::new(0.7, N - 1, Some(1.0), None),
      #[cfg(feature = "malliavin")]
      None,
    );

    plot_1d!(fbm.sample(), "Fractional Brownian Motion (H = 0.7)");
  }

  #[test]
  #[cfg(feature = "malliavin")]
  fn fbm_malliavin() {
    let fbm = FBM::new(
      0.7,
      N,
      Some(1.0),
      None,
      FGN::new(0.7, N - 1, Some(1.0), None),
      Some(true),
    );
    let process = fbm.sample();
    let malliavin = fbm.malliavin();
    plot_2d!(
      process,
      "Fractional Brownian Motion (H = 0.7)",
      malliavin,
      "Malliavin derivative of Fractional Brownian Motion (H = 0.7)"
    );
  }
}
