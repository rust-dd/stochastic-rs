use impl_new_derive::ImplNew;
use ndarray::Array1;

use crate::stochastic::{noise::fgn::FGN, SamplingExt};

#[derive(ImplNew)]
pub struct FJacobi {
  pub alpha: f64,
  pub beta: f64,
  pub sigma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub fgn: FGN,
  #[cfg(feature = "cuda")]
  #[default(false)]
  cuda: bool,
}

impl FJacobi {
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

impl SamplingExt<f64> for FJacobi {
  /// Sample the Fractional Jacobi process
  fn sample(&self) -> Array1<f64> {
    assert!(self.alpha > 0.0, "alpha must be positive");
    assert!(self.beta > 0.0, "beta must be positive");
    assert!(self.sigma > 0.0, "sigma must be positive");
    assert!(self.alpha < self.beta, "alpha must be less than beta");

    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let fgn = self.fgn();

    let mut fjacobi = Array1::<f64>::zeros(self.n);
    fjacobi[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      fjacobi[i] = match fjacobi[i - 1] {
        _ if fjacobi[i - 1] <= 0.0 && i > 0 => 0.0,
        _ if fjacobi[i - 1] >= 1.0 && i > 0 => 1.0,
        _ => {
          fjacobi[i - 1]
            + (self.alpha - self.beta * fjacobi[i - 1]) * dt
            + self.sigma * (fjacobi[i - 1] * (1.0 - fjacobi[i - 1])).sqrt() * fgn[i - 1]
        }
      }
    }

    fjacobi
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
  fn fjacobi_length_equals_n() {
    let fjacobi = FJacobi::new(
      0.43,
      0.5,
      0.8,
      N,
      Some(X0),
      Some(1.0),
      None,
      FGN::new(0.7, N - 1, Some(1.0), None),
    );

    assert_eq!(fjacobi.sample().len(), N);
  }

  #[test]
  fn fjacobi_starts_with_x0() {
    let fjacobi = FJacobi::new(
      0.43,
      0.5,
      0.8,
      N,
      Some(X0),
      Some(1.0),
      None,
      FGN::new(0.7, N - 1, Some(1.0), None),
    );

    assert_eq!(fjacobi.sample()[0], X0);
  }

  #[test]
  fn fjacobi_plot() {
    let fjacobi = FJacobi::new(
      0.43,
      0.5,
      0.8,
      N,
      Some(X0),
      Some(1.0),
      None,
      FGN::new(0.7, N - 1, Some(1.0), None),
    );

    plot_1d!(fjacobi.sample(), "Fractional Jacobi process");
  }

  #[test]
  #[ignore = "Not implemented"]
  #[cfg(feature = "malliavin")]
  fn fjacobi_malliavin() {
    unimplemented!();
  }
}
