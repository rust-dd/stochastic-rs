use ndarray::Array1;

use crate::stochastic::c;
use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

/// Gompertz diffusion
/// dX_t = (a - b ln X_t) X_t dt + sigma X_t dW_t
pub struct Gompertz<T: Float> {
  pub a: T,
  pub b: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: Float> Gompertz<T> {
  pub fn new(a: T, b: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      a,
      b,
      sigma,
      n,
      x0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: Float> Process<T> for Gompertz<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = &self.gn.sample();

    let mut x = Array1::<T>::zeros(self.n);
    let threshold = c(1e-12);
    x[0] = self.x0.unwrap_or(T::zero()).max(threshold);

    for i in 1..self.n {
      let xi = x[i - 1].max(threshold);
      let drift = (self.a - self.b * xi.ln()) * xi * dt;
      let diff = self.sigma * xi * gn[i - 1];
      let next = xi + drift + diff;
      x[i] = next.max(threshold);
    }

    x
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::plot_1d;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn gompertz_length_equals_n() {
    let proc = Gompertz::new(1.0, 0.5, 0.3, N, Some(X0), Some(1.0));
    assert_eq!(proc.sample().len(), N);
  }

  #[test]
  fn gompertz_starts_with_x0() {
    let proc = Gompertz::new(1.0, 0.5, 0.3, N, Some(X0), Some(1.0));
    assert_eq!(proc.sample()[0], X0.max(1e-12));
  }

  #[test]
  fn gompertz_plot() {
    let proc = Gompertz::new(1.0, 0.5, 0.3, N, Some(X0), Some(1.0));
    plot_1d!(proc.sample(), "Gompertz diffusion");
  }
}
