use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand::Rng;
use rand_distr::{Exp, Uniform};
use scilib::math::basic::gamma;

use crate::stochastic::{process::poisson::Poisson, SamplingExt};

/// RDTS process (Rapidly Decreasing Tempered Stable process)
/// https://sci-hub.se/https://doi.org/10.1016/j.jbankfin.2010.01.015
///
#[derive(ImplNew)]
pub struct RDTS<T> {
  /// Positive jump rate lambda_plus (corresponds to G)
  pub lambda_plus: T, // G
  /// Negative jump rate lambda_minus (corresponds to M)
  pub lambda_minus: T, // M
  /// Jump activity parameter alpha (corresponds to Y), with 0 < alpha < 2
  pub alpha: T,
  /// Number of time steps
  pub n: usize,
  /// Jumps
  pub j: usize,
  /// Initial value
  pub x0: Option<T>,
  /// Total time horizon
  pub t: Option<T>,
  /// Number of samples for parallel sampling (not used in this implementation)
  pub m: Option<usize>,
}

impl SamplingExt<f64> for RDTS<f64> {
  fn sample(&self) -> Array1<f64> {
    let mut rng = rand::thread_rng();

    let t_max = self.t.unwrap_or(1.0);
    let dt = t_max / (self.n - 1) as f64;
    let mut x = Array1::<f64>::zeros(self.n);
    x[0] = self.x0.unwrap_or(0.0);

    let C = (gamma(2.0 - self.alpha)
      * (self.lambda_plus.powf(self.alpha - 2.0) + self.lambda_minus.powf(self.alpha - 2.0)))
    .powi(-1);

    let b_t = -C
      * (gamma((1.0 - self.alpha) / 2.0) / 2.0_f64.powf((self.alpha + 1.0) / 2.0))
      * (self.lambda_plus.powf(self.alpha - 1.0) - self.lambda_minus.powf(self.alpha - 1.0));

    let U = Array1::<f64>::random(self.j, Uniform::new(0.0, 1.0));
    let E = Array1::<f64>::random(self.j, Exp::new(1.0).unwrap());
    let tau = Array1::<f64>::random(self.j, Uniform::new(0.0, 1.0));
    let poisson = Poisson::new(1.0, Some(self.j), None, None);
    let poisson = poisson.sample();

    for i in 1..self.n {
      let mut jump_component = 0.0;
      let t_1 = (i - 1) as f64 * dt;
      let t = i as f64 * dt;

      for j in 1..self.j {
        if tau[j] > t_1 && tau[j] <= t {
          let v_j = if rng.gen_bool(0.5) {
            self.lambda_plus
          } else {
            -self.lambda_minus
          };

          let term1 = (self.alpha * poisson[j] / (2.0 * C * t_max)).powf(-1.0 / self.alpha);
          let term2 = 0.5 * E[j].powf(0.5) * U[j].powf(1.0 / self.alpha) / v_j.abs();
          let jump_size = term1.min(term2) * (v_j / v_j.abs());

          jump_component += jump_size;
        }
      }

      x[i] = x[i - 1] + jump_component + b_t * dt;
    }

    x
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Axis;

  use super::*;
  use crate::{plot_1d, plot_nd, stochastic::N};

  #[test]
  fn rdts_length_equals_n() {
    let cgmy = RDTS::new(5.0, 5.0, 0.7, N, 1000, Some(0.0), Some(1.0), None);
    assert_eq!(cgmy.sample().len(), N);
  }

  #[test]
  fn rdts_starts_with_x0() {
    let cgmy = RDTS::new(5.0, 5.0, 0.7, N, 1000, Some(0.0), Some(1.0), None);
    assert_eq!(cgmy.sample()[0], 0.0);
  }

  #[test]
  fn rdts_plot() {
    let cgmy = RDTS::new(25.46, 4.604, 0.52, 100, 1024, Some(2.0), Some(1.0), None);
    plot_1d!(cgmy.sample(), "RDTS Process");
  }

  #[test]
  fn rdts_plot_multi() {
    let cgmy = RDTS::new(25.46, 4.604, 0.52, N, 10000, Some(2.0), Some(1.0), Some(10));
    plot_nd!(cgmy.sample_par(), "RDTS Process");
  }
}
