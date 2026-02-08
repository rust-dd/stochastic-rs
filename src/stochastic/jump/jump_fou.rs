use impl_new_derive::ImplNew;
use ndarray::s;
use ndarray::Array1;
use rand_distr::Distribution;

use crate::stochastic::noise::fgn::FGN;
use crate::stochastic::process::cpoisson::CompoundPoisson;
use crate::stochastic::Sampling3DExt;
use crate::stochastic::SamplingExt;

#[derive(ImplNew)]
pub struct JumpFOU<D, T>
where
  D: Distribution<T> + Send + Sync,
{
  pub mu: T,
  pub sigma: T,
  pub theta: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub m: Option<usize>,
  pub fgn: FGN<T>,
  pub cpoisson: CompoundPoisson<D, T>,
  #[cfg(feature = "cuda")]
  #[default(false)]
  cuda: bool,
}

impl<D> JumpFOU<D, f64>
where
  D: Distribution<f64> + Send + Sync,
{
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

impl<D> SamplingExt<f64> for JumpFOU<D, f64>
where
  D: Distribution<f64> + Send + Sync,
{
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let fgn = self.fgn();
    let mut jump_fou = Array1::<f64>::zeros(self.n);
    jump_fou[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      let [.., jumps] = self.cpoisson.sample();

      jump_fou[i] = jump_fou[i - 1]
        + self.theta * (self.mu - jump_fou[i - 1]) * dt
        + self.sigma * fgn[i - 1]
        + jumps.sum();
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

impl<D> JumpFOU<D, f32>
where
  D: Distribution<f32> + Send + Sync,
{
  fn fgn(&self) -> Array1<f32> {
    self.fgn.sample()
  }
}

impl<D> SamplingExt<f32> for JumpFOU<D, f32>
where
  D: Distribution<f32> + Send + Sync,
{
  fn sample(&self) -> Array1<f32> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let fgn = self.fgn();
    let mut jump_fou = Array1::<f32>::zeros(self.n);
    jump_fou[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      let [.., jumps] = self.cpoisson.sample();

      jump_fou[i] = jump_fou[i - 1]
        + self.theta * (self.mu - jump_fou[i - 1]) * dt
        + self.sigma * fgn[i - 1]
        + jumps.sum();
    }

    jump_fou.slice(s![..self.n()]).to_owned()
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
      N,
      Some(X0),
      Some(1.0),
      None,
      FGN::<f64>::new(0.7, N, None, None),
      CompoundPoisson::new(
        None,
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64), None),
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
      N,
      Some(X0),
      Some(1.0),
      None,
      FGN::<f64>::new(0.7, N, None, None),
      CompoundPoisson::new(
        None,
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64), None),
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
      N,
      Some(X0),
      Some(1.0),
      None,
      FGN::<f64>::new(0.7, N, None, None),
      CompoundPoisson::new(
        None,
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64), None),
      ),
    );

    plot_1d!(jump_fou.sample(), "Jump FOU process");
  }

  #[test]
  #[ignore = "Not implemented"]
  #[cfg(feature = "malliavin")]
  fn jump_fou_malliavin() {
    unimplemented!()
  }
}
