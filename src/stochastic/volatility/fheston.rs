use ndarray::Array1;
use statrs::function::gamma::gamma;

use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct RoughHeston<T: FloatExt> {
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

impl<T: FloatExt> RoughHeston<T> {
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

impl<T: FloatExt> ProcessExt<T> for RoughHeston<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = self.gn.sample();
    let mut yt = Array1::<T>::zeros(self.n);
    let mut zt = Array1::<T>::zeros(self.n);
    let mut sigma_tilde2 = Array1::<T>::zeros(self.n);
    let mut v2 = Array1::zeros(self.n);

    let v0_sq = self.v0.unwrap_or(T::one()).powi(2);
    yt[0] = v0_sq;
    zt[0] = T::zero();
    sigma_tilde2[0] = v0_sq;
    v2[0] = v0_sq;
    let g = gamma(self.hurst.to_f64().unwrap() - 0.5);

    for i in 1..self.n {
      let t = dt * T::from_usize_(i);
      yt[i] = self.theta + (yt[i - 1] - self.theta) * (-self.kappa * dt).exp();
      zt[i] = zt[i - 1] * (-self.kappa * dt).exp()
        + sigma_tilde2[i - 1].max(T::zero()).sqrt() * gn[i - 1];

      // CIR process: sigma_tilde^2 = Y_t + nu * Z_t
      sigma_tilde2[i] = yt[i] + self.nu * zt[i];

      let integral = (0..i)
        .map(|j| {
          let tj = T::from_usize_(j) * dt;
          ((t - tj).powf(self.hurst - T::from_f64_fast(0.5)) * zt[j]) * dt
        })
        .sum::<T>();

      v2[i] = yt[i]
        + self.c1.unwrap_or(T::one()) * self.nu * zt[i]
        + self.c2.unwrap_or(T::one()) * self.nu * integral / T::from_f64_fast(g);
    }

    v2
  }
}
