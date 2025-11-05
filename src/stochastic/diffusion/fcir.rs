use impl_new_derive::ImplNew;
use ndarray::Array1;

use crate::stochastic::{noise::fgn::FGN, SamplingExt};

/// Fractional Cox-Ingersoll-Ross (FCIR) process.
/// dX(t) = theta(mu - X(t))dt + sigma * sqrt(X(t))dW^H(t)
/// where X(t) is the FCIR process.
#[derive(ImplNew)]
pub struct FCIR<T> {
  pub theta: f64,
  pub mu: f64,
  pub sigma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub use_sym: Option<bool>,
  pub m: Option<usize>,
  pub fgn: FGN<T>,
  #[cfg(feature = "cuda")]
  #[default(false)]
  cuda: bool,
}

impl FCIR<f64> {
  fn fgn(&self) -> Array1<f64> {
    #[cfg(feature = "cuda")]
    if self.cuda {
      if self.m.is_some() && self.m.unwrap() > 1 {
        panic!("m must be None or 1 when using CUDA");
      }

      return self.fgn.sample_cuda().unwrap().left().unwrap();
    }

    self.fgn.sample()
  }
}

impl SamplingExt<f64> for FCIR<f64> {
  /// Sample the Fractional Cox-Ingersoll-Ross (FCIR) process
  fn sample(&self) -> Array1<f64> {
    assert!(
      2.0 * self.theta * self.mu >= self.sigma.powi(2),
      "2 * theta * mu < sigma^2"
    );

    let fgn = self.fgn();
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;

    let mut fcir = Array1::<f64>::zeros(self.n);
    fcir[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      let dfcir = self.theta * (self.mu - fcir[i - 1]) * dt
        + self.sigma * (fcir[i - 1]).abs().sqrt() * fgn[i - 1];

      fcir[i] = match self.use_sym.unwrap_or(false) {
        true => (fcir[i - 1] + dfcir).abs(),
        false => (fcir[i - 1] + dfcir).max(0.0),
      };
    }

    fcir
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }

  #[cfg(feature = "cuda")]
  fn set_cuda(&mut self, cuda: bool) {
    self.cuda = cuda;
  }
}

#[cfg(test)]
mod tests {
  use crate::{
    plot_1d,
    stochastic::{noise::fgn::FGN, SamplingExt, N, X0},
  };

  use super::*;

  #[test]
  fn fcir_length_equals_n() {
    let fcir = FCIR::new(
      1.0,
      1.2,
      0.2,
      N,
      Some(X0),
      Some(1.0),
      Some(false),
      Some(1),
      FGN::new(0.7, N, Some(1.0), None),
    );

    assert_eq!(fcir.sample().len(), N);
  }

  #[test]
  fn fcir_starts_with_x0() {
    let fcir = FCIR::new(
      1.0,
      1.2,
      0.2,
      N,
      Some(X0),
      Some(1.0),
      Some(false),
      Some(1),
      FGN::new(0.7, N, Some(1.0), None),
    );

    assert_eq!(fcir.sample()[0], X0);
  }

  #[test]
  fn fcir_plot() {
    let fcir = FCIR::new(
      1.0,
      1.2,
      0.2,
      N,
      Some(X0),
      Some(1.0),
      Some(false),
      Some(1),
      FGN::new(0.7, N, Some(1.0), None),
    );

    plot_1d!(
      fcir.sample(),
      "Fractional Cox-Ingersoll-Ross (FCIR) process"
    );
  }

  #[test]
  #[ignore = "Not implemented"]
  #[cfg(feature = "malliavin")]
  fn fcir_malliavin() {
    unimplemented!();
  }
}
