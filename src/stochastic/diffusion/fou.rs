use ndarray::{Array1, Array2, Axis, Zip};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::stochastic::{noise::fgn::FGN, SamplingExt};

pub struct FOU<T> {
  pub hurst: T,
  pub theta: T,
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub m: Option<usize>,
  fgn: FGN<T>,
  #[cfg(feature = "cuda")]
  cuda: bool,
}

#[cfg(feature = "f64")]
impl FOU<f64> {
  #[must_use]
  pub fn new(
    hurst: f64,
    theta: f64,
    mu: f64,
    sigma: f64,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    m: Option<usize>,
  ) -> Self {
    let fgn = FGN::<f64>::new(hurst, n - 1, t, m);

    Self {
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      m,
      fgn,
      #[cfg(feature = "cuda")]
      cuda: false,
    }
  }

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

#[cfg(feature = "f64")]
impl SamplingExt<f64> for FOU<f64> {
  /// Sample the Fractional Ornstein-Uhlenbeck (FOU) process
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let fgn = self.fgn();

    let mut fou = Array1::<f64>::zeros(self.n);
    fou[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      fou[i] = fou[i - 1] + self.theta * (self.mu - fou[i - 1]) * dt + self.sigma * fgn[i - 1];
    }

    fou
  }

  /// Sample the Fractional Ornstein-Uhlenbeck (FOU) process in parallel
  fn sample_par(&self) -> ndarray::Array2<f64> {
    let n = self.n();
    let m = self.m().unwrap();
    let dt = self.t.unwrap_or(1.0) / (n - 1) as f64;
    let mut xs = Array2::zeros((m, n));
    let fgn = self.fgn.sample_par();

    debug_assert_eq!(fgn.nrows(), m);
    debug_assert_eq!(fgn.ncols(), n - 1);

    Zip::from(xs.axis_iter_mut(Axis(0)))
      .and(fgn.axis_iter(Axis(0)))
      .into_par_iter()
      .for_each(|(mut fou, fgn)| {
        fou[0] = self.x0.unwrap_or(0.0);

        for i in 1..n {
          fou[i] = fou[i - 1] + self.theta * (self.mu - fou[i - 1]) * dt + self.sigma * fgn[i - 1];
        }
      });

    xs
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

#[cfg(feature = "f32")]
impl FOU<f32> {
  #[must_use]
  pub fn new(
    hurst: f32,
    theta: f32,
    mu: f32,
    sigma: f32,
    n: usize,
    x0: Option<f32>,
    t: Option<f32>,
    m: Option<usize>,
  ) -> Self {
    let fgn = FGN::<f32>::new(hurst, n - 1, t, m);

    Self {
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      m,
      fgn,
      #[cfg(feature = "cuda")]
      cuda: false,
    }
  }

  fn fgn(&self) -> Array1<f32> {
    self.fgn.sample()
  }
}

#[cfg(feature = "f32")]
impl SamplingExt<f32> for FOU<f32> {
  /// Sample the Fractional Ornstein-Uhlenbeck (FOU) process
  fn sample(&self) -> Array1<f32> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let fgn = self.fgn();

    let mut fou = Array1::<f32>::zeros(self.n);
    fou[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      fou[i] = fou[i - 1] + self.theta * (self.mu - fou[i - 1]) * dt + self.sigma * fgn[i - 1];
    }

    fou
  }

  /// Sample the Fractional Ornstein-Uhlenbeck (FOU) process in parallel
  fn sample_par(&self) -> ndarray::Array2<f32> {
    let n = self.n();
    let m = self.m().unwrap();
    let dt = self.t.unwrap_or(1.0) / (n - 1) as f32;
    let mut xs = Array2::zeros((m, n));
    let fgn = self.fgn.sample_par();

    debug_assert_eq!(fgn.nrows(), m);
    debug_assert_eq!(fgn.ncols(), n - 1);

    Zip::from(xs.axis_iter_mut(Axis(0)))
      .and(fgn.axis_iter(Axis(0)))
      .into_par_iter()
      .for_each(|(mut fou, fgn)| {
        fou[0] = self.x0.unwrap_or(0.0);

        for i in 1..n {
          fou[i] = fou[i - 1] + self.theta * (self.mu - fou[i - 1]) * dt + self.sigma * fgn[i - 1];
        }
      });

    xs
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
  use crate::{
    plot_1d,
    stochastic::{SamplingExt, N, X0},
  };

  use super::*;

  #[test]
  fn fou_length_equals_n() {
    let fou = FOU::<f64>::new(0.7, 2.0, 1.0, 0.8, N, Some(X0), Some(1.0), None);

    assert_eq!(fou.sample().len(), N);
  }

  #[test]
  fn fou_starts_with_x0() {
    let fou = FOU::<f64>::new(0.7, 2.0, 1.0, 0.8, N, Some(X0), Some(1.0), None);

    assert_eq!(fou.sample()[0], X0);
  }

  #[test]
  fn fou_plot() {
    let fou = FOU::<f64>::new(0.7, 2.0, 1.0, 0.8, N, Some(X0), Some(1.0), None);

    plot_1d!(fou.sample(), "Fractional Ornstein-Uhlenbeck (FOU) Process");
  }

  #[test]
  #[ignore = "Not implemented"]
  #[cfg(feature = "malliavin")]
  fn fou_malliavin() {
    unimplemented!();
  }
}
