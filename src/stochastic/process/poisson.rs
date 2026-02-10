use ndarray::Array0;
use ndarray::Array1;
use ndarray::Axis;
use ndarray::Dim;
use ndarray_rand::rand_distr::Distribution;
#[cfg(not(feature = "simd"))]
use ndarray_rand::rand_distr::Exp;
use ndarray_rand::RandomExt;
use rand::rng;

#[cfg(feature = "simd")]
use crate::distributions::exp::SimdExp;
use crate::stochastic::Float;
use crate::stochastic::Process;

#[derive(Clone, Copy)]
pub struct Poisson<T: Float> {
  pub lambda: T,
  pub n: Option<usize>,
  pub t_max: Option<T>,
}

impl<T: Float> Poisson<T> {
  pub fn new(lambda: T, n: Option<usize>, t_max: Option<T>) -> Self {
    Poisson { lambda, n, t_max }
  }
}

impl<T: Float> Process<T> for Poisson<T> {
  type Output = Array1<T>;
  type Noise = Self;

  fn sample(&self) -> Self::Output {
    #[cfg(not(feature = "simd"))]
    let distr = Exp::new(T::one() / self.lambda).unwrap();
    #[cfg(feature = "simd")]
    let distr = SimdExp::new(T::one() / self.lambda);

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

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    self.sample()
  }

  fn euler_maruyama(
    &self,
    _noise_fn: impl Fn(&Self::Noise) -> <Self::Noise as Process<T>>::Output,
  ) -> Self::Output {
    unimplemented!()
  }
}
