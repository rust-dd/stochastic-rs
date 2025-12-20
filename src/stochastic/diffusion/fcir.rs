use ndarray::{Array1, Array2, Axis, Zip};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::stochastic::{noise::fgn::FGN, SamplingExt};

/// Fractional Cox-Ingersoll-Ross (FCIR) process.
/// dX(t) = theta(mu - X(t))dt + sigma * sqrt(X(t))dW^H(t)
/// where X(t) is the FCIR process.
pub struct FCIR<T> {
  pub hurst: T,
  pub theta: T,
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub use_sym: Option<bool>,
  pub m: Option<usize>,
  fgn: FGN<T>,
  #[cfg(feature = "cuda")]
  cuda: bool,
}

#[cfg(feature = "f64")]
impl FCIR<f64> {
  #[must_use]
  pub fn new(
    hurst: f64,
    theta: f64,
    mu: f64,
    sigma: f64,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    use_sym: Option<bool>,
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
      use_sym,
      m,
      fgn,
      #[cfg(feature = "cuda")]
      cuda: false,
    }
  }
}

#[cfg(feature = "f64")]
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

#[cfg(feature = "f64")]
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

  /// Sample the Fractional Cox-Ingersoll-Ross (FCIR) process in parallel
  fn sample_par(&self) -> ndarray::Array2<f64> {
    assert!(
      2.0 * self.theta * self.mu >= self.sigma.powi(2),
      "2 * theta * mu < sigma^2"
    );

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
      .for_each(|(mut fcir, fgn)| {
        fcir[0] = self.x0.unwrap_or(0.0);

        for i in 1..n {
          let dfcir = self.theta * (self.mu - fcir[i - 1]) * dt
            + self.sigma * fcir[i - 1].abs().sqrt() * fgn[i - 1];

          fcir[i] = match self.use_sym.unwrap_or(false) {
            true => (fcir[i - 1] + dfcir).abs(),
            false => (fcir[i - 1] + dfcir).max(0.0),
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

#[cfg(feature = "f32")]
impl FCIR<f32> {
  #[must_use]
  pub fn new(
    hurst: f32,
    theta: f32,
    mu: f32,
    sigma: f32,
    n: usize,
    x0: Option<f32>,
    t: Option<f32>,
    use_sym: Option<bool>,
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
      use_sym,
      m,
      fgn,
      #[cfg(feature = "cuda")]
      cuda: false,
    }
  }
}

#[cfg(feature = "f32")]
impl FCIR<f32> {
  fn fgn(&self) -> Array1<f32> {
    self.fgn.sample()
  }
}

#[cfg(feature = "f32")]
impl SamplingExt<f32> for FCIR<f32> {
  fn sample(&self) -> Array1<f32> {
    assert!(
      2.0 * self.theta * self.mu >= self.sigma.powi(2),
      "2 * theta * mu < sigma^2"
    );

    let fgn = self.fgn();
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;

    let mut fcir = Array1::<f32>::zeros(self.n);
    fcir[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      let dfcir = (self.theta * (self.mu - fcir[i - 1]) * dt)
        + (self.sigma * (fcir[i - 1]).abs().sqrt()) * fgn[i - 1];

      fcir[i] = match self.use_sym.unwrap_or(false) {
        true => (fcir[i - 1] + dfcir).abs(),
        false => (fcir[i - 1] + dfcir).max(0.0),
      };
    }

    fcir
  }

  /// Sample the Fractional Cox-Ingersoll-Ross (FCIR) process in parallel
  fn sample_par(&self) -> ndarray::Array2<f32> {
    assert!(
      2.0 * self.theta * self.mu >= self.sigma.powi(2),
      "2 * theta * mu < sigma^2"
    );

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
      .for_each(|(mut fcir, fgn)| {
        fcir[0] = self.x0.unwrap_or(0.0);

        for i in 1..n {
          let dfcir = self.theta * (self.mu - fcir[i - 1]) * dt
            + self.sigma * fcir[i - 1].abs().sqrt() * fgn[i - 1];

          fcir[i] = match self.use_sym.unwrap_or(false) {
            true => (fcir[i - 1] + dfcir).abs(),
            false => (fcir[i - 1] + dfcir).max(0.0),
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
  use crate::{
    plot_1d,
    stochastic::{SamplingExt, N, X0},
  };

  use super::*;

  #[test]
  fn fcir_length_equals_n() {
    let fcir = FCIR::<f64>::new(
      0.7,
      1.0,
      1.2,
      0.2,
      N,
      Some(X0),
      Some(1.0),
      Some(false),
      Some(1),
    );

    assert_eq!(fcir.sample().len(), N);
  }

  #[test]
  fn fcir_starts_with_x0() {
    let fcir = FCIR::<f64>::new(
      0.7,
      1.0,
      1.2,
      0.2,
      N,
      Some(X0),
      Some(1.0),
      Some(false),
      Some(1),
    );

    assert_eq!(fcir.sample()[0], X0);
  }

  #[test]
  fn fcir_plot() {
    let fcir = FCIR::<f64>::new(
      0.7,
      1.0,
      1.2,
      0.2,
      N,
      Some(X0),
      Some(1.0),
      Some(false),
      Some(1),
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
