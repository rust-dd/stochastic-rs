use ndarray::Array1;
use rand_distr::Distribution;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::process::cpoisson::CompoundPoisson;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct LevyDiffusion<T, D>
where
  T: Float,
  D: Distribution<T> + Send + Sync,
{
  pub gamma: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub cpoisson: CompoundPoisson<T, D>,
  gn: Gn<T>,
}

impl<T, D> LevyDiffusion<T, D>
where
  T: Float,
  D: Distribution<T> + Send + Sync,
{
  fn new(
    gamma: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    cpoisson: CompoundPoisson<T, D>,
  ) -> Self {
    Self {
      gamma,
      sigma,
      n,
      x0,
      t,
      cpoisson,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T, D> Process<T> for LevyDiffusion<T, D>
where
  T: Float,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = &self.gn.sample();

    let mut levy = Array1::<T>::zeros(self.n);
    levy[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      let [.., jumps] = self.cpoisson.sample();
      levy[i] = levy[i - 1] + self.gamma * dt + self.sigma * gn[i - 1] + jumps.sum();
    }

    levy
  }
}

#[cfg(test)]
mod tests {
  use rand_distr::Normal;

  use super::*;
  use crate::plot_1d;
  use crate::stochastic::process::poisson::Poisson;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn levy_diffusion_length_equals_n() {
    let levy = LevyDiffusion::new(
      2.25,
      2.5,
      N,
      Some(X0),
      Some(1.0),
      CompoundPoisson::new(
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64)),
      ),
    );

    assert_eq!(levy.sample().len(), N);
  }

  #[test]
  fn levy_diffusion_starts_with_x0() {
    let levy = LevyDiffusion::new(
      2.25,
      2.5,
      N,
      Some(X0),
      Some(1.0),
      CompoundPoisson::new(
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64)),
      ),
    );

    assert_eq!(levy.sample()[0], X0);
  }

  #[test]
  fn levy_diffusion_plot() {
    let levy = LevyDiffusion::new(
      2.25,
      2.5,
      N,
      Some(X0),
      Some(1.0),
      CompoundPoisson::new(
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64)),
      ),
    );

    plot_1d!(levy.sample(), "Levy diffusion process");
  }
}
