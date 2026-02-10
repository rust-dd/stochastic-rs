use ndarray::Array0;
use ndarray::Array1;
use ndarray::Axis;
use ndarray::Dim;
use ndarray_rand::RandomExt;
use rand::rng;
use rand_distr::Distribution;

use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct CustomJt<T, D>
where
  T: Float,
  D: Distribution<T> + Send + Sync,
{
  pub n: Option<usize>,
  pub t_max: Option<T>,
  pub distribution: D,
}

impl<T, D> CustomJt<T, D>
where
  T: Float,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(n: Option<usize>, t_max: Option<T>, distribution: D) -> Self {
    CustomJt {
      n,
      t_max,
      distribution,
    }
  }
}

impl<T, D> Process<T> for CustomJt<T, D>
where
  T: Float,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    if let Some(n) = self.n {
      let random = Array1::random(n, &self.distribution);
      let mut x = Array1::<T>::zeros(n);
      for i in 1..n {
        x[i] = x[i - 1] + random[i - 1];
      }

      x
    } else if let Some(t_max) = self.t_max {
      let mut x = Array1::from(vec![T::zero()]);
      let mut t = T::zero();

      while t < t_max {
        t += self.distribution.sample(&mut rng());
        x.push(Axis(0), Array0::from_elem(Dim(()), t).view())
          .unwrap();
      }

      x
    } else {
      panic!("n or t_max must be provided");
    }
  }
}
