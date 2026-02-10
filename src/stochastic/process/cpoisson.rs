use ndarray::Array1;
use ndarray::Axis;
use rand::rng;
use rand_distr::Distribution;

use super::poisson::Poisson;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct CompoundPoisson<T, D>
where
  T: Float,
  D: Distribution<T> + Send + Sync,
{
  pub distribution: D,
  pub poisson: Poisson<T>,
}

impl<T, D> CompoundPoisson<T, D>
where
  T: Float,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(distribution: D, poisson: Poisson<T>) -> Self {
    Self {
      distribution,
      poisson,
    }
  }
}

impl<T, D> Process<T> for CompoundPoisson<T, D>
where
  T: Float,
  D: Distribution<T> + Send + Sync,
{
  type Output = [Array1<T>; 3];
  type Noise = Poisson<T>;

  fn sample(&self) -> Self::Output {
    self.euler_maruyama(|p| p.sample())
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    self.euler_maruyama(|p| p.sample_simd())
  }

  fn euler_maruyama(
    &self,
    noise_fn: impl Fn(&Self::Noise) -> <Self::Noise as Process<T>>::Output,
  ) -> Self::Output {
    let poisson = noise_fn(&self.poisson);
    let mut jumps = Array1::<T>::zeros(poisson.len());
    for i in 1..poisson.len() {
      jumps[i] = self.distribution.sample(&mut rng());
    }

    let mut cum_jupms = jumps.clone();
    cum_jupms.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

    [poisson, cum_jupms, jumps]
  }
}
