use ndarray::Array1;
use ndarray::Axis;
use rand::rng;
use rand_distr::Distribution;

use super::customjt::CustomJt;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct CompoundCustom<T, D1, D2>
where
  T: Float,
  D1: Distribution<T> + Send + Sync,
  D2: Distribution<T> + Send + Sync,
{
  pub n: Option<usize>,
  pub t_max: Option<T>,
  pub m: Option<usize>,
  pub jumps_distribution: D1,
  pub jump_times_distribution: D2,
  pub customjt: CustomJt<T, D2>,
}

impl<T, D1, D2> CompoundCustom<T, D1, D2>
where
  T: Float,
  D1: Distribution<T> + Send + Sync,
  D2: Distribution<T> + Send + Sync,
{
  pub fn new(
    n: Option<usize>,
    t_max: Option<T>,
    m: Option<usize>,
    jumps_distribution: D1,
    jump_times_distribution: D2,
    customjt: CustomJt<T, D2>,
  ) -> Self {
    if n.is_none() && t_max.is_none() {
      panic!("n or t_max must be provided");
    }

    Self {
      n,
      t_max,
      m,
      jumps_distribution,
      jump_times_distribution,
      customjt,
    }
  }
}

impl<T, D1, D2> Process<T> for CompoundCustom<T, D1, D2>
where
  T: Float,
  D1: Distribution<T> + Send + Sync,
  D2: Distribution<T> + Send + Sync,
{
  type Output = [Array1<T>; 3];

  fn sample(&self) -> Self::Output {
    let p = self.customjt.sample();
    let mut jumps = Array1::<T>::zeros(self.n.unwrap_or(p.len()));
    for i in 1..p.len() {
      jumps[i] = self.jumps_distribution.sample(&mut rng());
    }

    let mut cum_jupms = jumps.clone();
    cum_jupms.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

    [p, cum_jupms, jumps]
  }
}
