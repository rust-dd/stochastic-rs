use ndarray::Array1;
use statrs::function::gamma::gamma;

use crate::f;
use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct RoughHeston<T: Float> {
  pub hurst: T,
  pub v0: Option<T>,
  pub theta: T,
  pub kappa: T,
  pub nu: T,
  pub c1: Option<T>,
  pub c2: Option<T>,
  pub t: Option<T>,
  pub n: usize,
  gn: Gn<T>,
}

impl<T: Float> RoughHeston<T> {
  pub fn new(
    hurst: T,
    v0: Option<T>,
    theta: T,
    kappa: T,
    nu: T,
    c1: Option<T>,
    c2: Option<T>,
    t: Option<T>,
    n: usize,
  ) -> Self {
    RoughHeston {
      hurst,
      v0,
      theta,
      kappa,
      nu,
      c1,
      c2,
      t,
      n,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: Float> Process<T> for RoughHeston<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = self.gn.sample();
    let mut yt = Array1::<T>::zeros(self.n);
    let mut zt = Array1::<T>::zeros(self.n);
    let mut v2 = Array1::zeros(self.n);

    yt[0] =
      self.theta + (self.v0.unwrap_or(f!(1)).powi(2) - self.theta) * (-self.kappa * f!(0)).exp();
    zt[0] = f!(0); // Initial condition for Z_t, typically 0 for such integrals.
    v2[0] = self.v0.unwrap_or(f!(1)).powi(2);

    for i in 1..self.n {
      let t = dt * f!(i);
      yt[i] = self.theta + (yt[i - 1] - self.theta) * (-self.kappa * dt).exp();
      zt[i] = zt[i - 1] * (-self.kappa * dt).exp() + (v2[i - 1].powi(2)).sqrt() * gn[i - 1];

      let integral = (0..i)
        .map(|j| {
          let tj = f!(j) * dt;
          ((t - tj).powf(self.hurst - f!(0.5)) * zt[j]) * dt
        })
        .sum::<T>();

      v2[i] = yt[i]
        + self.c1.unwrap_or(f!(1)) * self.nu * zt[i]
        + self.c2.unwrap_or(f!(1)) * self.nu * integral
          / f!(gamma(self.hurst.to_f64().unwrap() + 0.5));
    }

    v2
  }
}
