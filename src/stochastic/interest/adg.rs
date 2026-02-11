use ndarray::Array1;
use ndarray::Array2;

use crate::f;
use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

/// Ahn-Dittmar-Gallant (ADG) model
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
  gn: Gn<T>,
}

impl<T: Float> ADG<T> {
  pub fn new(
    k: fn(T) -> T,
    theta: fn(T) -> T,
    sigma: Array1<T>,
    phi: fn(T) -> T,
    b: fn(T) -> T,
    c: fn(T) -> T,
    n: usize,
    xn: usize,
    x0: Array1<T>,
    t: Option<T>,
  ) -> Self {
    Self {
      k,
      theta,
      sigma,
      phi,
      b,
      c,
      n,
      xn,
      x0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: Float> Process<T> for ADG<T> {
  type Output = Array2<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();

    let mut adg = Array2::<T>::zeros((self.xn, self.n));
    for i in 0..self.xn {
      adg[(i, 0)] = self.x0[i];
    }

    for i in 0..self.xn {
      let gn = &self.gn.sample();

      for j in 1..self.n {
        let t = j as f64 * dt;
        adg[(i, j)] = adg[(i, j - 1)]
          + ((self.k)(t) - (self.theta)(t) * adg[(i, j - 1)]) * dt
          + self.sigma[i] * gn[j - 1];
      }
    }

    let mut r = Array2::zeros((self.xn, self.n));

    for i in 0..self.xn {
      let phi = Array1::<T>::from_shape_fn(self.n, |j| (self.phi)(f!(j) * dt));
      let b = Array1::<T>::from_shape_fn(self.n, |j| (self.b)(f!(j) * dt));
      let c = Array1::<T>::from_shape_fn(self.n, |j| (self.c)(f!(j) * dt));

      r.row_mut(i)
        .assign(&(phi + b * adg.row(i).t().to_owned() * c * adg.row(i)));
    }

    r
  }
}
