use ndarray::Array1;
use rand_distr::Distribution;

use crate::stochastic::noise::fgn::FGN;
use crate::stochastic::Float;
use crate::stochastic::ProcessExt;

pub struct JumpFOUCustom<T, D>
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
  pub jump_times: D,
  pub jump_sizes: D,
  fgn: FGN<T>,
}

impl<T, D> JumpFOUCustom<T, D>
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
    jump_times: D,
    jump_sizes: D,
  ) -> Self {
    Self {
      hurst,
      mu,
      sigma,
      theta,
      n,
      x0,
      t,
      jump_times,
      jump_sizes,
      fgn: FGN::new(hurst, n - 1, t),
    }
  }
}

impl<T, D> ProcessExt<T> for JumpFOUCustom<T, D>
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
    let mut jump_times = Array1::<T>::zeros(self.n);
    jump_times.mapv_inplace(|_| self.jump_times.sample(&mut rand::rng()));

    for i in 1..self.n {
      let t = T::from_usize_(i) * dt;
      // check if t is a jump time
      let mut jump = T::zero();
      if jump_times[i] < t && t - dt <= jump_times[i] {
        jump = self.jump_sizes.sample(&mut rand::rng());
      }

      jump_fou[i] = jump_fou[i - 1]
        + self.theta * (self.mu - jump_fou[i - 1]) * dt
        + self.sigma * fgn[i - 1]
        + jump;
    }

    jump_fou
  }
}
