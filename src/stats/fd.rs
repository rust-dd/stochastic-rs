//! # Fd
//!
//! $$
//! D=2-\frac{1}{p}\frac{\log V_p(2)-\log V_p(1)}{\log 2}
//! $$
//!
use std::f64::consts::LN_2;

use linreg::linear_regression;
use ndarray::Array1;

/// Fractal dimension.
pub struct FractalDim {
  /// Observed data/sample vector.
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
    if p <= 0.0 {
      panic!("p must be positive for variogram-based fractal dimension.");
    }
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
    if v1 <= 0.0 || v2 <= 0.0 || !v1.is_finite() || !v2.is_finite() {
      panic!("Variogram is undefined for degenerate/non-finite path increments.");
    }

    2.0 - (1.0 / p) * ((v2.ln() - v1.ln()) / LN_2)
  }

  /// Calculate the Higuchi fractal dimension of the path.
  pub fn higuchi_fd(&self, kmax: usize) -> f64 {
    let n_times = self.x.len();
    if n_times < 3 {
      panic!("A path must have at least 3 points for Higuchi fractal dimension.");
    }
    if kmax < 2 {
      panic!("kmax must be at least 2 for Higuchi fractal dimension.");
    }

    let k_upper = kmax.min(n_times - 1);
    let mut x_reg = Array1::<f64>::zeros(k_upper);
    let mut y_reg = Array1::<f64>::zeros(k_upper);
    let mut used = 0usize;

    for k in 1..=k_upper {
      let mut lm_sum = 0.0;
      let mut lm_count = 0usize;

      for m in 0..k {
        let n_max = (n_times - m - 1) / k;
        if n_max == 0 {
          continue;
        }

        let mut ll = 0.0;
        for j in 1..=n_max {
          ll += (self.x[m + j * k] - self.x[m + (j - 1) * k]).abs();
        }

        ll /= k as f64;
        ll *= (n_times - 1) as f64 / (k * n_max) as f64;
        if ll.is_finite() && ll > 0.0 {
          lm_sum += ll;
          lm_count += 1;
        }
      }

      if lm_count > 0 {
        let lk = lm_sum / lm_count as f64;
        if lk.is_finite() && lk > 0.0 {
          x_reg[used] = (1.0 / k as f64).ln();
          y_reg[used] = lk.ln();
          used += 1;
        }
      }
    }

    if used < 2 {
      panic!("Not enough valid scales for Higuchi regression.");
    }
    let x = &x_reg.as_slice().unwrap()[..used];
    let y = &y_reg.as_slice().unwrap()[..used];
    let (slope, _) = linear_regression(x, y).unwrap();
    slope
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;

  use super::FractalDim;
  use crate::stochastic::process::fbm::FBM;
  use crate::traits::ProcessExt;

  #[test]
  fn variogram_fbm_matches_theory_on_average() {
    let h = 0.72_f64;
    let d_theory = 2.0 - h;
    let n = 4096_usize;
    let m = 192_usize;
    let fbm = FBM::new(h, n, Some(1.0));

    let mut d_sum = 0.0;
    for _ in 0..m {
      let x = fbm.sample();
      let fd = FractalDim::new(x);
      d_sum += fd.variogram(Some(2.0));
    }
    let d_est = d_sum / m as f64;
    assert!(
      (d_est - d_theory).abs() < 0.05,
      "variogram FD mismatch: D_est={d_est}, D={d_theory}"
    );
  }

  #[test]
  fn higuchi_fbm_matches_theory_on_average() {
    let h = 0.72_f64;
    let d_theory = 2.0 - h;
    let n = 4096_usize;
    let kmax = 32_usize;
    let m = 96_usize;
    let fbm = FBM::new(h, n, Some(1.0));

    let mut d_sum = 0.0;
    for _ in 0..m {
      let x = fbm.sample();
      let fd = FractalDim::new(x);
      d_sum += fd.higuchi_fd(kmax);
    }
    let d_est = d_sum / m as f64;
    assert!(
      (d_est - d_theory).abs() < 0.05,
      "Higuchi FD mismatch: D_est={d_est}, D={d_theory}"
    );
  }

  #[test]
  #[should_panic(expected = "Variogram is undefined")]
  fn variogram_rejects_degenerate_path() {
    let x = Array1::from_elem(128, 1.0);
    let fd = FractalDim::new(x);
    let _ = fd.variogram(Some(2.0));
  }
}
