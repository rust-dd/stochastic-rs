use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand::Rng;
use scilib::math::basic::gamma;

use crate::distributions::exp::SimdExp;
use crate::distributions::uniform::SimdUniform;
use crate::stochastic::process::poisson::Poisson;
use crate::stochastic::FloatExt;
use crate::stochastic::ProcessExt;

/// RDTS process (Rapidly Decreasing Tempered Stable process)
/// https://sci-hub.se/https://doi.org/10.1016/j.jbankfin.2010.01.015
pub struct RDTS<T: FloatExt> {
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

impl<T: FloatExt> RDTS<T> {
  /// Create a new RDTS process
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

impl<T: FloatExt> ProcessExt<T> for RDTS<T> {
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

    let g = gamma((1.0 - self.alpha.to_f64().unwrap()) / 2.0);
    let b_t = -C
      * (T::from_f64_fast(g) / T::from_usize_(2).powf((self.alpha + T::one()) / T::from_usize_(2)))
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

          let divisor = T::from_usize_(2) * C * t_max;
          let numerator = self.alpha * P[j];
          let term1 = (numerator / divisor).powf(-T::one() / self.alpha);
          let term2 = T::from_f64_fast(0.5)
            * E[j].powf(T::from_f64_fast(0.5))
            * U[j].powf(T::one() / self.alpha)
            / v_j.abs();
          let jump_size = term1.min(term2) * (v_j / v_j.abs());

          jump_component += jump_size;
        }
      }

      x[i] = x[i - 1] + jump_component + b_t * dt;
    }

    x
  }
}
