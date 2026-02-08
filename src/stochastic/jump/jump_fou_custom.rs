use impl_new_derive::ImplNew;
use ndarray::s;
use ndarray::Array1;
use rand_distr::Distribution;

use crate::stochastic::noise::fgn::FGN;
use crate::stochastic::SamplingExt;

#[derive(ImplNew)]
pub struct JumpFOUCustom<D, T>
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
  pub jump_times: D,
  pub jump_sizes: D,
  #[cfg(feature = "cuda")]
  #[default(false)]
  cuda: bool,
}

impl<D> JumpFOUCustom<D, f64>
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

impl<D> SamplingExt<f64> for JumpFOUCustom<D, f64>
where
  D: Distribution<f64> + Send + Sync,
{
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let fgn = self.fgn();
    let mut jump_fou = Array1::<f64>::zeros(self.n);
    jump_fou[0] = self.x0.unwrap_or(0.0);
    let mut jump_times = Array1::<f64>::zeros(self.n);
    jump_times.mapv_inplace(|_| self.jump_times.sample(&mut rand::rng()));

    for i in 1..self.n {
      let t = i as f64 * dt;
      // check if t is a jump time
      let mut jump = 0.0;
      if jump_times[i] < t && t - dt <= jump_times[i] {
        jump = self.jump_sizes.sample(&mut rand::rng());
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

  #[cfg(feature = "cuda")]
  fn set_cuda(&mut self, cuda: bool) {
    self.cuda = cuda;
  }
}

impl<D> JumpFOUCustom<D, f32>
where
  D: Distribution<f32> + Send + Sync,
{
  fn fgn(&self) -> Array1<f32> {
    self.fgn.sample()
  }
}

impl<D> SamplingExt<f32> for JumpFOUCustom<D, f32>
where
  D: Distribution<f32> + Send + Sync,
{
  fn sample(&self) -> Array1<f32> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let fgn = self.fgn();
    let mut jump_fou = Array1::<f32>::zeros(self.n);
    jump_fou[0] = self.x0.unwrap_or(0.0);
    let mut jump_times = Array1::<f32>::zeros(self.n);
    jump_times.mapv_inplace(|_| self.jump_times.sample(&mut rand::rng()));

    for i in 1..self.n {
      let t = i as f32 * dt;
      let mut jump = 0.0;
      if jump_times[i] < t && t - dt <= jump_times[i] {
        jump = self.jump_sizes.sample(&mut rand::rng());
      }

      jump_fou[i] = jump_fou[i - 1]
        + self.theta * (self.mu - jump_fou[i - 1]) * dt
        + self.sigma * fgn[i - 1]
        + jump;
    }

    jump_fou.slice(s![..self.n()]).to_owned()
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }

  #[cfg(feature = "cuda")]
  fn set_cuda(&mut self, cuda: bool) {
    self.cuda = cuda;
  }
}

#[cfg(test)]
mod tests {
  use rand_distr::Gamma;
  use rand_distr::Weibull;

  use super::*;
  use crate::plot_1d;
  use crate::stochastic::N;
  use crate::stochastic::X0;

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
      FGN::<f64>::new(0.7, N, None, None),
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
      FGN::<f64>::new(0.7, N, None, None),
      Gamma::new(1.0, 1.0).unwrap(),
      Gamma::new(12.0, 1.0).unwrap(),
    );

    plot_1d!(jump_fou.sample(), "Jump FOU Gamma process");
  }
}
