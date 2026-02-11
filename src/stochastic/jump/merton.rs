use ndarray::Array1;
use rand_distr::Distribution;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::process::cpoisson::CompoundPoisson;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct Merton<T, D>
where
  T: Float,
  D: Distribution<T> + Send + Sync,
{
  pub alpha: T,
  pub sigma: T,
  pub lambda: T,
  pub theta: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub cpoisson: CompoundPoisson<T, D>,
  gn: Gn<T>,
}

impl<T, D> Merton<T, D>
where
  T: Float,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(
    alpha: T,
    sigma: T,
    lambda: T,
    theta: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    cpoisson: CompoundPoisson<T, D>,
  ) -> Self {
    Self {
      alpha,
      sigma,
      lambda,
      theta,
      n,
      x0,
      t,
      cpoisson,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T, D> Process<T> for Merton<T, D>
where
  T: Float,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = &self.gn.sample();

    let mut merton = Array1::<T>::zeros(self.n);
    merton[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      let [.., jumps] = self.cpoisson.sample();

      merton[i] = merton[i - 1]
        + (self.alpha
          - self.sigma.powf(T::from_usize(2).unwrap()) / T::from_usize(2).unwrap()
          - self.lambda * self.theta)
          * dt
        + self.sigma * gn[i - 1]
        + jumps.sum();
    }

    merton
  }
}

#[cfg(test)]
mod tests {
  use rand_distr::Normal;

  use super::*;
  use crate::plot_1d;
  use crate::stochastic::process::poisson::Poisson;
  use crate::stochastic::N;
  use crate::stochastic::S0;
  use crate::stochastic::X0;

  #[test]
  fn merton_length_equals_n() {
    let merton = Merton::new(
      2.25,
      2.5,
      1.0,
      1.0,
      N,
      Some(X0),
      Some(1.0),
      CompoundPoisson::new(
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64)),
      ),
    );

    assert_eq!(merton.sample().len(), N);
  }

  #[test]
  fn merton_starts_with_x0() {
    let merton = Merton::new(
      2.25,
      2.5,
      1.0,
      1.0,
      N,
      Some(X0),
      Some(1.0),
      CompoundPoisson::new(
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64)),
      ),
    );

    assert_eq!(merton.sample()[0], X0);
  }

  #[test]
  fn merton_plot() {
    let merton = Merton::new(
      2.25,
      2.5,
      1.0,
      1.0,
      N,
      Some(S0),
      Some(1.0),
      CompoundPoisson::new(
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64)),
      ),
    );

    plot_1d!(merton.sample(), "Merton process");
  }
}
