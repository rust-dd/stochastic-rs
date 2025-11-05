use impl_new_derive::ImplNew;
use ndarray::{Array0, Array1, Axis, Dim};
use ndarray_rand::rand_distr::{Distribution, Exp};
use ndarray_rand::RandomExt;
use rand::thread_rng;

use crate::stochastic::SamplingExt;

#[derive(ImplNew)]
pub struct Poisson<T> {
  pub lambda: T,
  pub n: Option<usize>,
  pub t_max: Option<T>,
  pub m: Option<usize>,
}

impl SamplingExt<f64> for Poisson<f64> {
  fn sample(&self) -> Array1<f64> {
    if let Some(n) = self.n {
      let exponentials = Array1::random(n, Exp::new(1.0 / self.lambda).unwrap());
      let mut poisson = Array1::<f64>::zeros(n);
      for i in 1..n {
        poisson[i] = poisson[i - 1] + exponentials[i - 1];
      }

      poisson
    } else if let Some(t_max) = self.t_max {
      let mut poisson = Array1::from(vec![0.0]);
      let mut t = 0.0;

      while t < t_max {
        t += Exp::new(1.0 / self.lambda)
          .unwrap()
          .sample(&mut thread_rng());

        if t < t_max {
          poisson
            .push(Axis(0), Array0::from_elem(Dim(()), t).view())
            .unwrap();
        }
      }

      poisson
    } else {
      panic!("n or t_max must be provided");
    }
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n.unwrap_or(0)
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}
