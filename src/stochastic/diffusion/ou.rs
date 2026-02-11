use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

#[derive(Clone, Copy)]
pub struct OU<T: Float> {
  pub theta: T,
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub gn: Gn<T>,
}

impl<T: Float> OU<T> {
  pub fn new(theta: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    OU {
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: Float> Process<T> for OU<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = self.gn.sample();

    let mut ou = Array1::<T>::zeros(self.n);
    ou[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      ou[i] = ou[i - 1] + self.theta * (self.mu - ou[i - 1]) * dt + self.sigma * gn[i - 1]
    }

    ou
  }
}

#[cfg(test)]
mod tests {
  use super::OU;
  use crate::plot_1d;
  use crate::stochastic::Process;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn ou_length_equals_n() {
    let ou = OU::new(2.0, 1.0, 0.8, N, Some(X0), Some(1.0));

    assert_eq!(ou.sample().len(), N);
  }

  #[test]
  fn ou_starts_with_x0() {
    let ou = OU::new(2.0, 1.0, 0.8, N, Some(X0), Some(1.0));

    assert_eq!(ou.sample()[0], X0);
  }

  #[test]
  fn ou_plot() {
    let ou = OU::new(2.0, 1.0, 0.8, N, Some(X0), Some(1.0));

    plot_1d!(ou.sample(), "Fractional Ornstein-Uhlenbeck (FOU) Process");
  }
}
