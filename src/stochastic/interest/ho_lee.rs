use std::sync::Arc;

use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::FloatExt;
use crate::stochastic::ProcessExt;

#[allow(non_snake_case)]
pub struct HoLee<T: FloatExt> {
  pub f_T: Option<Arc<dyn Fn(T) -> T + Send + Sync + 'static>>,
  pub theta: Option<T>,
  pub sigma: T,
  pub n: usize,
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: FloatExt> HoLee<T> {
  pub fn new(
    f_T: Option<Arc<dyn Fn(T) -> T + Send + Sync + 'static>>,
    theta: Option<T>,
    sigma: T,
    n: usize,
    t: Option<T>,
  ) -> Self {
    assert!(
      theta.is_some() || f_T.is_some(),
      "theta or f_T must be provided"
    );

    Self {
      f_T,
      theta,
      sigma,
      n,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for HoLee<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = &self.gn.sample();

    let mut r = Array1::<T>::zeros(self.n);

    for i in 1..self.n {
      let t = T::from_usize_(i) * dt;
      let drift = if let Some(r#fn) = self.f_T.as_ref() {
        (r#fn)(t) + self.sigma.powf(T::from_usize_(2)) * t
      } else {
        self.theta.unwrap() + self.sigma.powf(T::from_usize_(2)) * t
      };

      r[i] = r[i - 1] + drift * dt + self.sigma * gn[i - 1];
    }

    r
  }
}
