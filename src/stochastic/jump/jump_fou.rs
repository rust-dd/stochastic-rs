use ndarray::Array1;
use rand_distr::Distribution;

use crate::stochastic::noise::fgn::FGN;
use crate::stochastic::process::cpoisson::CompoundPoisson;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct JumpFOU<T, D>
where
  T: Float,
  D: Distribution<T> + Send + Sync,
{
  pub hurst: T,
  pub theta: T,
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub cpoisson: CompoundPoisson<T, D>,
  fgn: FGN<T>,
}

impl<T, D> JumpFOU<T, D>
where
  T: Float,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(
    hurst: T,
    theta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    cpoisson: CompoundPoisson<T, D>,
  ) -> Self {
    Self {
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      cpoisson,
      fgn: FGN::new(hurst, n - 1, t),
    }
  }
}

impl<T, D> Process<T> for JumpFOU<T, D>
where
  T: Float,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.fgn.dt();
    let fgn = &self.fgn.sample();

    let mut jump_fou = Array1::<T>::zeros(self.n);
    jump_fou[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      let [.., jumps] = self.cpoisson.sample();

      jump_fou[i] = jump_fou[i - 1]
        + self.theta * (self.mu - jump_fou[i - 1]) * dt
        + self.sigma * fgn[i - 1]
        + jumps.sum();
    }

    jump_fou
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
  fn jump_fou_length_equals_n() {
    let jump_fou = JumpFOU::new(
      2.25,
      2.5,
      1.0,
      1.0,
      N,
      Some(X0),
      None,
      CompoundPoisson::new(
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64)),
      ),
    );

    assert_eq!(jump_fou.sample().len(), N);
  }

  #[test]
  fn jump_fou_starts_with_x0() {
    let jump_fou = JumpFOU::new(
      2.25,
      2.5,
      1.0,
      1.0,
      N,
      Some(X0),
      None,
      CompoundPoisson::new(
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64)),
      ),
    );

    assert_eq!(jump_fou.sample()[0], X0);
  }

  #[test]
  fn jump_fou_plot() {
    let jump_fou = JumpFOU::new(
      2.25,
      2.5,
      1.0,
      1.0,
      N,
      Some(X0),
      None,
      CompoundPoisson::new(
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64)),
      ),
    );

    plot_1d!(jump_fou.sample(), "Jump FOU process");
  }
}
