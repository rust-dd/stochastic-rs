use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand::Rng;
#[cfg(not(feature = "simd"))]
use rand_distr::Exp;
#[cfg(not(feature = "simd"))]
use rand_distr::Uniform;
use scilib::math::basic::gamma;

#[cfg(feature = "simd")]
use crate::distributions::exp::SimdExp;
#[cfg(feature = "simd")]
use crate::distributions::uniform::SimdUniform;
use crate::stochastic::process::poisson::Poisson;
use crate::stochastic::Float;
use crate::stochastic::Process;

/// CTS process (Classical Tempered Stable process)
/// https://sci-hub.se/https://doi.org/10.1016/j.jbankfin.2010.01.015
///
pub struct CTS<T: Float> {
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
}

impl<T: Float> CTS<T> {
  /// Create a new CTS process
  pub fn new(
    lambda_plus: T,
    lambda_minus: T,
    alpha: T,
    n: usize,
    j: usize,
    x0: Option<T>,
    t: Option<T>,
  ) -> Self {
    Self {
      lambda_plus,
      lambda_minus,
      alpha,
      n,
      j,
      x0,
      t,
    }
  }
}

impl<T: Float> Process<T> for CTS<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut rng = rand::rng();

    let t_max = self.t.unwrap_or(T::one());
    let dt = t_max / T::from_usize_(self.n - 1);
    let mut x = Array1::<T>::zeros(self.n);
    x[0] = self.x0.unwrap_or(T::zero());

    let g = gamma(2.0 - self.alpha.to_f64().unwrap());
    let C = (T::from_f64_fast(g)
      * (self.lambda_plus.powf(self.alpha - T::from_usize_(2))
        + self.lambda_minus.powf(self.alpha - T::from_usize_(2))))
    .powi(-1);

    let g = gamma(1.0 - self.alpha.to_f64().unwrap());
    let b_t = -C
      * T::from_f64_fast(g)
      * (self.lambda_plus.powf(self.alpha - T::one())
        - self.lambda_minus.powf(self.alpha - T::one()));

    let uniform = SimdUniform::new(T::zero(), T::one());
    let exp = SimdExp::new(T::one());

    let U = Array1::<T>::random(self.j, &uniform);
    let E = Array1::<T>::random(self.j, exp);
    let P = Poisson::new(T::one(), Some(self.j), None);
    let P = P.sample();
    let tau = Array1::<T>::random(self.j, &uniform);

    for i in 1..self.n {
      let mut jump_component = T::zero();
      let t_1 = T::from_usize_(i - 1) * dt;
      let t = T::from_usize_(i) * dt;

      for j in 1..self.j {
        if tau[j] > t_1 && tau[j] <= t {
          let v_j = if rng.random_bool(0.5) {
            self.lambda_plus
          } else {
            -self.lambda_minus
          };

          let numerator = self.alpha * P[j];
          let term1 = (numerator / C).powf(-T::one() / self.alpha);
          let term2 = E[j] * U[j].powf(T::one() / self.alpha) / v_j.abs();
          let jump_size = term1.min(term2) * (v_j / v_j.abs());

          jump_component += jump_size;
        }
      }

      x[i] = x[i - 1] + jump_component + b_t * dt;
    }

    x
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::plot_1d;
  use crate::stochastic::N;

  #[test]
  fn cts_length_equals_n() {
    let cts = CTS::new(5.0, 5.0, 0.7, N, 1000, Some(0.0), Some(1.0));
    assert_eq!(cts.sample().len(), N);
  }

  #[test]
  fn cts_starts_with_x0() {
    let cts = CTS::new(5.0, 5.0, 0.7, N, 1000, Some(0.0), Some(1.0));
    assert_eq!(cts.sample()[0], 0.0);
  }

  #[test]
  fn cts_plot() {
    let cts = CTS::new(25.46, 4.604, 0.52, N, 1024, Some(2.0), Some(1.0));
    plot_1d!(cts.sample(), "CTS Process");
  }
}
