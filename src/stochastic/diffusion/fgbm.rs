use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use ndarray::Zip;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

use crate::stochastic::noise::fgn::FGN;
use crate::stochastic::SamplingExt;

pub struct FGBM<T> {
  pub hurst: T,
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
impl FGBM<f64> {
  #[must_use]
  pub fn new(
    hurst: f64,
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
impl SamplingExt<f64> for FGBM<f64> {
  /// Sample the Fractional Geometric Brownian Motion (FGBM) process
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let fgn = self.fgn();

    let mut fgbm = Array1::<f64>::zeros(self.n);
    fgbm[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      fgbm[i] = fgbm[i - 1] + self.mu * fgbm[i - 1] * dt + self.sigma * fgbm[i - 1] * fgn[i - 1];
    }

    fgbm
  }

  /// Sample the Fractional Geometric Brownian Motion (FGBM) process in parallel
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
      .for_each(|(mut fgbm, fgn)| {
        fgbm[0] = self.x0.unwrap_or(0.0);

        for i in 1..n {
          fgbm[i] =
            fgbm[i - 1] + self.mu * fgbm[i - 1] * dt + self.sigma * fgbm[i - 1] * fgn[i - 1];
        }
      });

    xs
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

#[cfg(feature = "f32")]
impl FGBM<f32> {
  #[must_use]
  pub fn new(
    hurst: f32,
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
impl SamplingExt<f32> for FGBM<f32> {
  /// Sample the Fractional Geometric Brownian Motion (FGBM) process
  fn sample(&self) -> Array1<f32> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let fgn = self.fgn();

    let mut fgbm = Array1::<f32>::zeros(self.n);
    fgbm[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      fgbm[i] = fgbm[i - 1] + self.mu * fgbm[i - 1] * dt + self.sigma * fgbm[i - 1] * fgn[i - 1];
    }

    fgbm
  }

  /// Sample the Fractional Geometric Brownian Motion (FGBM) process in parallel
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
      .for_each(|(mut fgbm, fgn)| {
        fgbm[0] = self.x0.unwrap_or(0.0);

        for i in 1..n {
          fgbm[i] =
            fgbm[i - 1] + self.mu * fgbm[i - 1] * dt + self.sigma * fgbm[i - 1] * fgn[i - 1];
        }
      });

    xs
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
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::plot_1d;
  use crate::stochastic::SamplingExt;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn fgbm_length_equals_n() {
    let fgbm = FGBM::<f64>::new(0.7, 1.0, 0.8, N, Some(X0), Some(1.0), None);

    assert_eq!(fgbm.sample().len(), N);
  }

  #[test]
  fn fgbm_starts_with_x0() {
    let fgbm = FGBM::<f64>::new(0.7, 1.0, 0.8, N, Some(X0), Some(1.0), None);

    assert_eq!(fgbm.sample()[0], X0);
  }

  #[test]
  fn fgbm_plot() {
    let fgbm = FGBM::<f64>::new(0.7, 1.0, 0.8, N, Some(X0), Some(1.0), None);

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
