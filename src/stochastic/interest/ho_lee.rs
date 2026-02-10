use std::sync::Arc;

use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

#[allow(non_snake_case)]
pub struct HoLee<T: Float> {
  pub f_T: Option<Arc<dyn Fn(T) -> T + Send + Sync + 'static>>,
  pub theta: Option<T>,
  pub sigma: T,
  pub n: usize,
  pub t: Option<T>,
}

impl<T: Float> HoLee<T> {
  pub fn new(
    f_T: Option<Arc<dyn Fn(T) -> T + Send + Sync + 'static>>,
    theta: Option<T>,
    sigma: T,
    n: usize,
    t: Option<T>,
  ) -> Self {
    assert!(
      theta.is_none() && f_T.is_none(),
      "theta or f_T must be provided"
    );

    Self {
      f_T,
      theta,
      sigma,
      n,
      t,
    }
  }
}

impl<T: Float> Process<T> for HoLee<T> {
  type Output = Array1<T>;
  type Noise = Gn<T>;

  fn sample(&self) -> Self::Output {
    self.euler_maruyama(|gn| gn.sample())
  }

  fn sample_simd(&self) -> Self::Output {
    self.euler_maruyama(|gn| gn.sample_simd())
  }

  fn euler_maruyama(
    &self,
    noise_fn: impl Fn(&Self::Noise) -> <Self::Noise as Process<T>>::Output,
  ) -> Self::Output {
    let gn = Gn::new(self.n - 1, self.t);
    let dt = gn.dt();
    let gn = noise_fn(&gn);

    let mut r = Array1::<T>::zeros(self.n);

    for i in 1..self.n {
      let drift = if let Some(r#fn) = self.f_T.as_ref() {
        (r#fn)(i as f64 * dt) + self.sigma.powf(T::from_usize(2))
      } else {
        self.theta.unwrap() + self.sigma.powf(T::from_usize(2))
      };

      r[i] = r[i - 1] + drift * dt + self.sigma * gn[i - 1];
    }

    r
  }
}
