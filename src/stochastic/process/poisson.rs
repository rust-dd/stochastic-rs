use impl_new_derive::ImplNew;
use ndarray::Array0;
use ndarray::Array1;
use ndarray::Axis;
use ndarray::Dim;
use ndarray_rand::rand_distr::Distribution;
use ndarray_rand::rand_distr::Exp;
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

#[cfg(feature = "f64")]
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

#[cfg(feature = "f32")]
impl SamplingExt<f32> for Poisson<f32> {
  fn sample(&self) -> Array1<f32> {
    if let Some(n) = self.n {
      let exponentials = Array1::random(n, Exp::<f32>::new(1.0 / self.lambda).unwrap());
      let mut poisson = Array1::<f32>::zeros(n);
      for i in 1..n {
        poisson[i] = poisson[i - 1] + exponentials[i - 1];
      }

      poisson
    } else if let Some(t_max) = self.t_max {
      let mut poisson = Array1::from(vec![0.0f32]);
      let mut t = 0.0f32;

      while t < t_max {
        t += Exp::new((1.0 / self.lambda) as f64)
          .unwrap()
          .sample(&mut thread_rng()) as f32;

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

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f32> {
    use crate::stats::distr::exp::SimdExp;

    if let Some(n) = self.n {
      let exponentials = Array1::random(n, SimdExp::new(1.0 / self.lambda));
      let mut poisson = Array1::<f32>::zeros(n);
      for i in 1..n {
        poisson[i] = poisson[i - 1] + exponentials[i - 1];
      }

      poisson
    } else if let Some(_t_max) = self.t_max {
      // For t_max-based sampling, delegate to standard sample()
      // as dynamic array growth doesn't benefit from SIMD
      self.sample()
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
