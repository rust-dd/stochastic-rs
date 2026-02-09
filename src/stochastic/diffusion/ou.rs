use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct OU<T> {
  pub theta: T,
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
}

impl<T: Float> OU<T> {
  /// Create a new Ornstein-Uhlenbeck (OU) process
  pub fn new(theta: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
    }
  }
}

impl<T: Float> OU<T> {}

impl<T: Float> Process<T> for OU<T> {
  type Output = Array1<T>;
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
    let noise = noise_fn(&gn);

    let mut ou = Array1::<T>::zeros(self.n);
    ou[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      ou[i] = ou[i - 1] + self.theta * (self.mu - ou[i - 1]) * dt + self.sigma * noise[i - 1]
    }

    ou
  }
}

#[cfg(test)]
mod tests {
  use super::OU;
  use crate::plot_1d;
  use crate::stochastic::Process;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn ou_length_equals_n() {
    let ou = OU::new(2.0, 1.0, 0.8, N, Some(X0), Some(1.0));

    assert_eq!(ou.sample().len(), N);
  }

  #[test]
  fn ou_starts_with_x0() {
    let ou = OU::new(2.0, 1.0, 0.8, N, Some(X0), Some(1.0));

    assert_eq!(ou.sample()[0], X0);
  }

  #[test]
  fn ou_plot() {
    let ou = OU::new(2.0, 1.0, 0.8, N, Some(X0), Some(1.0));

    plot_1d!(ou.sample(), "Fractional Ornstein-Uhlenbeck (FOU) Process");
  }

  #[cfg(all(feature = "simd"))]
  #[test]
  fn sample_simd() {
    use std::time::Instant;

    let start = Instant::now();
    let ou = OU::new(2.0_f32, 1.0_f32, 0.8_f32, N, Some(X0 as f32), Some(1.0_f32));

    for _ in 0..100_000 {
      ou.sample_simd();
    }

    let elapsed = start.elapsed();
    println!("Elapsed time for sample_simd: {:?}", elapsed);

    let start = Instant::now();
    let ou = OU::new(2.0_f32, 1.0_f32, 0.8_f32, N, Some(X0 as f32), Some(1.0_f32));

    for _ in 0..100_000 {
      ou.sample();
    }

    let elapsed = start.elapsed();
    println!("Elapsed time for sample: {:?}", elapsed);
  }
}
