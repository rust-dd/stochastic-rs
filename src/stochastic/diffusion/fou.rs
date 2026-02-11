use ndarray::Array1;

use crate::stochastic::noise::fgn::FGN;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct FOU<T: Float> {
  pub hurst: T,
  pub theta: T,
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  fgn: FGN<T>,
}

impl<T: Float> FOU<T> {
  #[must_use]
  pub fn new(hurst: T, theta: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      fgn: FGN::new(hurst, n - 1, t),
    }
  }
}

impl<T: Float> Process<T> for FOU<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.fgn.dt();
    let fgn = self.fgn.sample();

    let mut fou = Array1::<T>::zeros(self.n);
    fou[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      fou[i] = fou[i - 1] + self.theta * (self.mu - fou[i - 1]) * dt + self.sigma * fgn[i - 1];
    }

    fou
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::plot_1d;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn fou_length_equals_n() {
    let fou = FOU::<f64>::new(0.7, 2.0, 1.0, 0.8, N, Some(X0), Some(1.0));
    assert_eq!(fou.sample().len(), N);
  }

  #[test]
  fn fou_starts_with_x0() {
    let fou = FOU::<f64>::new(0.7, 2.0, 1.0, 0.8, N, Some(X0), Some(1.0));
    assert_eq!(fou.sample()[0], X0);
  }

  #[test]
  fn fou_plot() {
    let fou = FOU::<f64>::new(0.7, 2.0, 1.0, 0.8, N, Some(X0), Some(1.0));
    plot_1d!(fou.sample(), "Fractional Ornstein-Uhlenbeck (FOU) Process");
  }
}
