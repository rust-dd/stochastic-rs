use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray::Array2;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

/// Ahn-Dittmar-Gallant (ADG) model
#[derive(ImplNew)]
pub struct ADG<T: Float> {
  pub k: fn(T) -> T,
  pub theta: fn(T) -> T,
  pub sigma: Array1<T>,
  pub phi: fn(T) -> T,
  pub b: fn(T) -> T,
  pub c: fn(T) -> T,
  pub n: usize,
  pub xn: usize,
  pub x0: Array1<T>,
  pub t: Option<T>,
}

impl<T: Float> Process<T> for ADG<T> {
  type Output = Array2<T>;
  type Noise = Gn<T>;

  fn sample(&self) -> Self::Output {
    self.euler_maruyama(|gn| gn.sample())
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    self.euler_maruyama(|gn| gn.sample_simd())
  }

  fn euler_maruyama(
    &self,
    noise_fn: impl Fn(&Self::Noise) -> <Self::Noise as Process<T>>::Output,
  ) -> Self::Output {
    let gn = Gn::new(self.n - 1, self.t);
    let dt = gn.dt();

    let mut adg = Array2::<T>::zeros((self.xn, self.n));
    for i in 0..self.xn {
      adg[(i, 0)] = self.x0[i];
    }

    for i in 0..self.xn {
      let gn = noise_fn(&gn);

      for j in 1..self.n {
        let t = j as f64 * dt;
        adg[(i, j)] = adg[(i, j - 1)]
          + ((self.k)(t) - (self.theta)(t) * adg[(i, j - 1)]) * dt
          + self.sigma[i] * gn[j - 1];
      }
    }

    let mut r = Array2::zeros((self.xn, self.n));

    for i in 0..self.xn {
      let phi = Array1::<T>::from_shape_fn(self.n, |j| (self.phi)(T::from_usize(j) * dt));
      let b = Array1::<T>::from_shape_fn(self.n, |j| (self.b)(T::from_usize(j) * dt));
      let c = Array1::<T>::from_shape_fn(self.n, |j| (self.c)(T::from_usize(j) * dt));

      r.row_mut(i)
        .assign(&(phi + b * adg.row(i).t().to_owned() * c * adg.row(i)));
    }

    r
  }
}
