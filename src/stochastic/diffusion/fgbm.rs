use ndarray::Array1;

use crate::stochastic::noise::fgn::FGN;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct FGBM<T: Float> {
  pub hurst: T,
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  fgn: FGN<T>,
}

impl<T: Float> FGBM<T> {
  #[must_use]
  pub fn new(hurst: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      hurst,
      mu,
      sigma,
      n,
      x0,
      t,
      fgn: FGN::new(hurst, n - 1, t),
    }
  }
}

impl<T: Float> Process<T> for FGBM<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.fgn.dt();
    let fgn = &self.fgn.sample();

    let mut fgbm = Array1::<T>::zeros(self.n);
    fgbm[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      fgbm[i] = fgbm[i - 1] + self.mu * fgbm[i - 1] * dt + self.sigma * fgbm[i - 1] * fgn[i - 1];
    }

    fgbm
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::plot_1d;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn fgbm_length_equals_n() {
    let fgbm = FGBM::<f64>::new(0.7, 1.0, 0.8, N, Some(X0), Some(1.0));

    assert_eq!(fgbm.sample().len(), N);
  }

  #[test]
  fn fgbm_starts_with_x0() {
    let fgbm = FGBM::<f64>::new(0.7, 1.0, 0.8, N, Some(X0), Some(1.0));

    assert_eq!(fgbm.sample()[0], X0);
  }

  #[test]
  fn fgbm_plot() {
    let fgbm = FGBM::<f64>::new(0.7, 1.0, 0.8, N, Some(X0), Some(1.0));

    plot_1d!(
      fgbm.sample(),
      "Fractional Geometric Brownian Motion (FGBM) process"
    );
  }
}
