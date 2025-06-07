use std::f64::consts::LN_2;

use linreg::linear_regression;
use ndarray::Array1;

/// Fractal dimension.
pub struct FractalDim {
  pub x: Array1<f64>,
}

impl FractalDim {
  #[must_use]
  pub fn new(x: Array1<f64>) -> Self {
    Self { x }
  }

  /// Calculate the variogram of the path.
  pub fn variogram(&self, p: Option<f64>) -> f64 {
    if self.x.len() < 3 {
      panic!("A path must have at least 3 points to calculate the variogram.");
    }

    let p = p.unwrap_or(1.0);
    let sum1: f64 = (1..self.x.len())
      .map(|i| (self.x[i] - self.x[i - 1]).abs().powf(p))
      .sum();
    let sum2: f64 = (2..self.x.len())
      .map(|i| (self.x[i] - self.x[i - 2]).abs().powf(p))
      .sum();

    let vp = |increments: f64, l: usize, x_len: usize| -> f64 {
      1.0 / (2.0 * (x_len - l) as f64) * increments
    };

    let v1 = vp(sum1, 1, self.x.len());
    let v2 = vp(sum2, 2, self.x.len());

    2.0 - (1.0 / p) * ((v2.ln() - v1.ln()) / LN_2)
  }

  /// Calculate the Higuchi fractal dimension of the path.
  pub fn higuchi_fd(&self, kmax: usize) -> f64 {
    let n_times = self.x.len();

    let mut lk = Array1::<f64>::zeros(kmax);
    let mut x_reg = Array1::<f64>::zeros(kmax);
    let mut y_reg = Array1::<f64>::zeros(kmax);

    for k in 1..=kmax {
      let mut lm = Array1::zeros(k);

      for m in 0..k {
        let mut ll = 0.0;
        let n_max = ((n_times - m - 1) as f64 / k as f64).floor() as usize;

        for j in 1..n_max {
          ll += (self.x[m + j * k] - self.x[m + (j - 1) * k]).abs();
        }

        ll /= k as f64;
        ll *= (n_times - 1) as f64 / (k * n_max) as f64;
        lm[m] = ll;
      }

      lk[k - 1] = lm.iter().sum::<f64>() / k as f64;
      x_reg[k - 1] = (1.0 / k as f64).ln();
      y_reg[k - 1] = lk[k - 1].ln();
    }

    let (slope, _) =
      linear_regression(x_reg.as_slice().unwrap(), y_reg.as_slice().unwrap()).unwrap();
    slope
  }
}

#[cfg(test)]
mod tests {
  use approx::assert_relative_eq;

  use crate::stochastic::{noise::fgn::FGN, process::fbm::FBM, SamplingExt, N};

  use super::*;

  #[test]
  fn test_variogram() {
    let hurst = 0.75;
    let x = FBM::new(hurst, N, None, None, FGN::new(hurst, N - 1, None, None));
    let fd = FractalDim::new(x.sample());
    let result = fd.variogram(None);
    assert_relative_eq!(2.0 - result, hurst, epsilon = 1e-1);
  }

  #[test]
  fn test_higuchi_fd() {
    let hurst = 0.75;
    let x = FBM::new(hurst, N, None, None, FGN::new(hurst, N - 1, None, None));
    let fd = FractalDim::new(x.sample());
    let result = fd.higuchi_fd(10);
    assert_relative_eq!(2.0 - result, hurst, epsilon = 1e-1);
  }
}
