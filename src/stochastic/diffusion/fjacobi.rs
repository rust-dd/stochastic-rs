use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use ndarray::Zip;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

use crate::stochastic::noise::fgn::FGN;
use crate::stochastic::SamplingExt;

pub struct FJacobi<T> {
  pub hurst: T,
  pub alpha: T,
  pub beta: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub m: Option<usize>,
  fgn: FGN<T>,
  #[cfg(feature = "cuda")]
  cuda: bool,
}

impl FJacobi<f64> {
  #[must_use]
  pub fn new(
    hurst: f64,
    alpha: f64,
    beta: f64,
    sigma: f64,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    m: Option<usize>,
  ) -> Self {
    let fgn = FGN::<f64>::new(hurst, n - 1, t, m);

    Self {
      hurst,
      alpha,
      beta,
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

impl SamplingExt<f64> for FJacobi<f64> {
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
      };
    }

    fjacobi
  }

  /// Sample the Fractional Jacobi process in parallel
  fn sample_par(&self) -> ndarray::Array2<f64> {
    assert!(self.alpha > 0.0, "alpha must be positive");
    assert!(self.beta > 0.0, "beta must be positive");
    assert!(self.sigma > 0.0, "sigma must be positive");
    assert!(self.alpha < self.beta, "alpha must be less than beta");

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
      .for_each(|(mut fjacobi, fgn)| {
        fjacobi[0] = self.x0.unwrap_or(0.0);

        for i in 1..n {
          fjacobi[i] = match fjacobi[i - 1] {
            _ if fjacobi[i - 1] <= 0.0 && i > 0 => 0.0,
            _ if fjacobi[i - 1] >= 1.0 && i > 0 => 1.0,
            _ => {
              fjacobi[i - 1]
                + (self.alpha - self.beta * fjacobi[i - 1]) * dt
                + self.sigma * (fjacobi[i - 1] * (1.0 - fjacobi[i - 1])).sqrt() * fgn[i - 1]
            }
          };
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

impl FJacobi<f32> {
  #[must_use]
  pub fn new(
    hurst: f32,
    alpha: f32,
    beta: f32,
    sigma: f32,
    n: usize,
    x0: Option<f32>,
    t: Option<f32>,
    m: Option<usize>,
  ) -> Self {
    let fgn = FGN::<f32>::new(hurst, n - 1, t, m);

    Self {
      hurst,
      alpha,
      beta,
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

impl SamplingExt<f32> for FJacobi<f32> {
  fn sample(&self) -> Array1<f32> {
    assert!(self.alpha > 0.0, "alpha must be positive");
    assert!(self.beta > 0.0, "beta must be positive");
    assert!(self.sigma > 0.0, "sigma must be positive");
    assert!(self.alpha < self.beta, "alpha must be less than beta");

    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let fgn = self.fgn();

    let mut fjacobi = Array1::<f32>::zeros(self.n);
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
      };
    }

    fjacobi
  }

  fn sample_par(&self) -> ndarray::Array2<f32> {
    assert!(self.alpha > 0.0, "alpha must be positive");
    assert!(self.beta > 0.0, "beta must be positive");
    assert!(self.sigma > 0.0, "sigma must be positive");
    assert!(self.alpha < self.beta, "alpha must be less than beta");

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
      .for_each(|(mut fjacobi, fgn)| {
        fjacobi[0] = self.x0.unwrap_or(0.0);

        for i in 1..n {
          fjacobi[i] = match fjacobi[i - 1] {
            _ if fjacobi[i - 1] <= 0.0 && i > 0 => 0.0,
            _ if fjacobi[i - 1] >= 1.0 && i > 0 => 1.0,
            _ => {
              fjacobi[i - 1]
                + (self.alpha - self.beta * fjacobi[i - 1]) * dt
                + self.sigma * (fjacobi[i - 1] * (1.0 - fjacobi[i - 1])).sqrt() * fgn[i - 1]
            }
          };
        }
      });

    xs
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
  use crate::plot_1d;
  use crate::stochastic::SamplingExt;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn fjacobi_length_equals_n() {
    let fjacobi = FJacobi::<f64>::new(0.7, 0.43, 0.5, 0.8, N, Some(X0), Some(1.0), None);

    assert_eq!(fjacobi.sample().len(), N);
  }

  #[test]
  fn fjacobi_starts_with_x0() {
    let fjacobi = FJacobi::<f64>::new(0.7, 0.43, 0.5, 0.8, N, Some(X0), Some(1.0), None);

    assert_eq!(fjacobi.sample()[0], X0);
  }

  #[test]
  fn fjacobi_plot() {
    let fjacobi = FJacobi::<f64>::new(0.7, 0.43, 0.5, 0.8, N, Some(X0), Some(1.0), None);

    plot_1d!(fjacobi.sample(), "Fractional Jacobi process");
  }

  #[test]
  #[ignore = "Not implemented"]
  fn fjacobi_malliavin() {
    unimplemented!();
  }
}
