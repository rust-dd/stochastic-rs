use ndarray::Array1;
use ndarray::Axis;
use rand::rng;
use rand_distr::Distribution;

use super::poisson::Poisson;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct CompoundPoisson<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub distribution: D,
  pub poisson: Poisson<T>,
}

impl<T, D> CompoundPoisson<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(distribution: D, poisson: Poisson<T>) -> Self {
    Self {
      distribution,
      poisson,
    }
  }
}

impl<T, D> ProcessExt<T> for CompoundPoisson<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = [Array1<T>; 3];

  fn sample(&self) -> Self::Output {
    let poisson = self.poisson.sample();
    let mut jumps = Array1::<T>::zeros(poisson.len());
    for i in 1..poisson.len() {
      jumps[i] = self.distribution.sample(&mut rng());
    }

    let mut cum_jupms = jumps.clone();
    cum_jupms.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

    [poisson, cum_jupms, jumps]
  }
}
