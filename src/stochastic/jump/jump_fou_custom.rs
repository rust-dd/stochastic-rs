use impl_new_derive::ImplNew;
use ndarray::{s, Array1};
use ndarray_rand::RandomExt;
use rand_distr::Distribution;

use crate::stochastic::{noise::fgn::FGN, Sampling};

#[derive(ImplNew)]
pub struct JumpFOUCustom<D>
where
  D: Distribution<f64> + Send + Sync,
{
  pub mu: f64,
  pub sigma: f64,
  pub theta: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub fgn: FGN,
  pub jump_times: D,
  pub jump_sizes: D,
}

impl<D> Sampling<f64> for JumpFOUCustom<D>
where
  D: Distribution<f64> + Send + Sync,
{
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let fgn = self.fgn.sample();
    let mut jump_fou = Array1::<f64>::zeros(self.n);
    jump_fou[0] = self.x0.unwrap_or(0.0);
    let mut jump_times = Array1::<f64>::zeros(self.n);
    jump_times.mapv_inplace(|_| self.jump_times.sample(&mut rand::thread_rng()));

    for i in 1..self.n {
      let t = i as f64 * dt;
      // check if t is a jump time
      let mut jump = 0.0;
      if jump_times[i] < t && t - dt <= jump_times[i] {
        jump = self.jump_sizes.sample(&mut rand::thread_rng());
      }

      jump_fou[i] = jump_fou[i - 1]
        + self.theta * (self.mu - jump_fou[i - 1]) * dt
        + self.sigma * fgn[i - 1]
        + jump;
    }

    jump_fou.slice(s![..self.n()]).to_owned()
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(test)]
mod tests {
  use rand_distr::{Gamma, Normal, Weibull};

  use crate::{
    plot_1d,
    stochastic::{process::poisson::Poisson, N, X0},
  };

  use super::*;

  #[test]
  fn jump_fou_weibull() {
    let jump_fou = JumpFOUCustom::new(
      2.25,
      2.5,
      1.0,
      N,
      Some(X0),
      Some(1.0),
      None,
      FGN::new(0.7, N, None, None),
      Weibull::new(4.0, 2.0).unwrap(),
      Weibull::new(10.0, 2.0).unwrap(),
    );

    plot_1d!(jump_fou.sample(), "Jump FOU Weibull process");
  }

  #[test]
  fn jump_fou_gamma() {
    let jump_fou = JumpFOUCustom::new(
      2.25,
      2.5,
      1.0,
      N,
      Some(X0),
      Some(1.0),
      None,
      FGN::new(0.7, N, None, None),
      Gamma::new(1.0, 1.0).unwrap(),
      Gamma::new(12.0, 1.0).unwrap(),
    );

    plot_1d!(jump_fou.sample(), "Jump FOU Gamma process");
  }
}
