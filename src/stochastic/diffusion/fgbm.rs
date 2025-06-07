use impl_new_derive::ImplNew;
use ndarray::Array1;

use crate::stochastic::{noise::fgn::FGN, SamplingExt};

#[derive(ImplNew)]
pub struct FGBM {
  pub mu: f64,
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

impl FGBM {
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

impl SamplingExt<f64> for FGBM {
  /// Sample the Fractional Geometric Brownian Motion (FGBM) process
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let fgn = self.fgn();

    let mut fgbm = Array1::<f64>::zeros(self.n);
    fgbm[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      fgbm[i] = fgbm[i - 1] + self.mu * fgbm[i - 1] * dt + self.sigma * fgbm[i - 1] * fgn[i - 1]
    }

    fgbm
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of paths
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
  fn fgbm_length_equals_n() {
    let fgbm = FGBM::new(
      1.0,
      0.8,
      N,
      Some(X0),
      Some(1.0),
      None,
      FGN::new(0.7, N - 1, Some(1.0), None),
    );

    assert_eq!(fgbm.sample().len(), N);
  }

  #[test]
  fn fgbm_starts_with_x0() {
    let fgbm = FGBM::new(
      1.0,
      0.8,
      N,
      Some(X0),
      Some(1.0),
      None,
      FGN::new(0.7, N - 1, Some(1.0), None),
    );

    assert_eq!(fgbm.sample()[0], X0);
  }

  #[test]
  fn fgbm_plot() {
    let fgbm = FGBM::new(
      1.0,
      0.8,
      N,
      Some(X0),
      Some(1.0),
      None,
      FGN::new(0.7, N - 1, Some(1.0), None),
    );

    plot_1d!(
      fgbm.sample(),
      "Fractional Geometric Brownian Motion (FGBM) process"
    );
  }

  #[test]
  #[ignore = "Not implemented"]
  #[cfg(feature = "malliavin")]
  fn fgbm_malliavin() {
    unimplemented!();
  }
}
