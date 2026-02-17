//! # Poisson
//!
//! $$
//! \mathbb{P}(N=k)=e^{-\lambda}\frac{\lambda^k}{k!},\ k\in\mathbb N_0
//! $$
//!
use ndarray::Array0;
use ndarray::Array1;
use ndarray::Axis;
use ndarray::Dim;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Distribution;
use rand::rng;

use crate::distributions::exp::SimdExp;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

#[derive(Clone, Copy)]
pub struct Poisson<T: FloatExt> {
  /// Jump intensity (expected arrivals per unit time).
  pub lambda: T,
  /// Optional fixed number of sampled events.
  /// If set, the process is generated with exactly `n` increments.
  pub n: Option<usize>,
  /// Optional terminal time for horizon-based sampling.
  /// If set (and `n` is `None`), events are sampled up to `t_max`.
  pub t_max: Option<T>,
}

impl<T: FloatExt> Poisson<T> {
  pub fn new(lambda: T, n: Option<usize>, t_max: Option<T>) -> Self {
    Poisson { lambda, n, t_max }
  }
}

impl<T: FloatExt> ProcessExt<T> for Poisson<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let distr = SimdExp::new(self.lambda);

    if let Some(n) = self.n {
      let exponentials = Array1::random(n, distr);
      let mut poisson = Array1::<T>::zeros(n);
      for i in 1..n {
        poisson[i] = poisson[i - 1] + exponentials[i - 1];
      }

      poisson
    } else if let Some(t_max) = self.t_max {
      let mut poisson = Array1::from(vec![T::zero()]);
      let mut t = T::zero();

      while t < t_max {
        t += distr.sample(&mut rng());

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
}

py_process_1d!(PyPoisson, Poisson,
  sig: (lambda_, n=None, t_max=None, dtype=None),
  params: (lambda_: f64, n: Option<usize>, t_max: Option<f64>)
);
