use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand::Rng;
use rand_distr::Exp;
use rand_distr::Uniform;
use scilib::math::basic::gamma;

use crate::stochastic::process::poisson::Poisson;
use crate::stochastic::SamplingExt;

/// CGMY process
///
/// The CGMY process is a pure jump Lévy process used in financial modeling to capture the dynamics
/// of asset returns with jumps and heavy tails. It is characterized by four parameters:
/// `C`, `G` (lambda_plus), `M` (lambda_minus), and `Y` (alpha).
///
/// The process is defined by the Lévy measure:
///
/// \[ \nu(x) = C \frac{e^{-G x}}{x^{1 + Y}} 1_{x > 0} + C \frac{e^{M x}}{|x|^{1 + Y}} 1_{x < 0} \]
///
/// where:
/// - `c` (C) > 0 controls the overall intensity of the jumps.
/// - `lambda_plus` (G) > 0 is the rate of exponential decay of positive jumps.
/// - `lambda_minus` (M) > 0 is the rate of exponential decay of negative jumps.
/// - `alfa` (Y), with 0 < `alfa` < 2, controls the jump activity (number of small jumps).
///
/// Series representation of the CGMY process:
/// \[ X(t) = \sum_{i=1}^{\infty} ((alpha * Gamma_j / 2C)^(-1/alpha) \land E_j * U_j^(1/alpha) * abs(V_j)^-1)) * V_j / |V_j| 1_[0, t] + b_t * t \]
///
/// This implementation simulates the CGMY process using a discrete approximation over a grid of time points.
/// At each time step, we generate a Poisson random number of jumps, and for each jump, we generate the jump size
/// according to the CGMY process. The process also includes a drift component computed from the parameters.
///
/// # References
///
/// - Cont, R., & Tankov, P. (2004). *Financial Modelling with Jump Processes*. Chapman and Hall/CRC.
/// - Madan, D. B., Carr, P., & Chang, E. C. (1998). The Variance Gamma Process and Option Pricing. *European Finance Review*, 2(1), 79-105.
/// https://www.econstor.eu/bitstream/10419/239493/1/175133161X.pdf
///
#[derive(ImplNew)]
pub struct CGMY<T> {
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

#[cfg(feature = "f64")]
impl SamplingExt<f64> for CGMY<f64> {
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
      * gamma(1.0 - self.alpha)
      * (self.lambda_plus.powf(self.alpha - 1.0) - self.lambda_minus.powf(self.alpha - 1.0));

    let U = Array1::<f64>::random(self.j, Uniform::new(0.0, 1.0));
    let E = Array1::<f64>::random(self.j, Exp::new(1.0).unwrap());
    let P = Poisson::new(1.0, Some(self.j), None, None);
    let P = P.sample();
    let tau = Array1::<f64>::random(self.j, Uniform::new(0.0, 1.0));

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

          let divisor: f64 = 2.0 * C * t_max;
          let numerator: f64 = self.alpha * P[j];
          let term1 = (numerator / divisor).powf(-1.0 / self.alpha);
          let term2 = E[j] * U[j].powf(1.0 / self.alpha) / v_j.abs();
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

#[cfg(feature = "f32")]
impl SamplingExt<f32> for CGMY<f32> {
  fn sample(&self) -> Array1<f32> {
    let mut rng = rand::thread_rng();

    let t_max = self.t.unwrap_or(1.0);
    let dt = t_max / (self.n - 1) as f32;
    let mut x = Array1::<f32>::zeros(self.n);
    x[0] = self.x0.unwrap_or(0.0);

    let C = (gamma(2.0 - self.alpha as f64)
      * (self.lambda_plus.powf(self.alpha - 2.0) as f64
        + self.lambda_minus.powf(self.alpha - 2.0) as f64)) as f32;
    let C = C.powi(-1);

    let b_t = -C
      * gamma(1.0 - self.alpha as f64) as f32
      * (self.lambda_plus.powf(self.alpha - 1.0) - self.lambda_minus.powf(self.alpha - 1.0));

    let U = Array1::random(self.j, Uniform::<f32>::new(0.0, 1.0));
    let E = Array1::random(self.j, Exp::<f32>::new(1.0).unwrap());
    let P = Poisson::<f32>::new(1.0, Some(self.j), None, None).sample();
    let tau = Array1::random(self.j, Uniform::<f32>::new(0.0, 1.0));

    for i in 1..self.n {
      let mut jump_component = 0.0;
      let t_1 = (i - 1) as f32 * dt;
      let t = i as f32 * dt;

      for j in 1..self.j {
        if tau[j] > t_1 && tau[j] <= t {
          let v_j = if rng.gen_bool(0.5) {
            self.lambda_plus
          } else {
            -self.lambda_minus
          };

          let divisor: f32 = 2.0 * C * t_max;
          let numerator: f32 = self.alpha * P[j];
          let term1 = (numerator / divisor).powf(-1.0 / self.alpha);
          let term2 = E[j] * U[j].powf(1.0 / self.alpha) / v_j.abs();
          let jump_size = term1.min(term2) * (v_j / v_j.abs());

          jump_component += jump_size;
        }
      }

      x[i] = x[i - 1] + jump_component + b_t * dt;
    }

    x
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f32> {
    use crate::stats::distr::exp::SimdExp;
    use crate::stats::distr::uniform::SimdUniform;

    let mut rng = rand::thread_rng();

    let t_max = self.t.unwrap_or(1.0);
    let dt = t_max / (self.n - 1) as f32;
    let mut x = Array1::<f32>::zeros(self.n);
    x[0] = self.x0.unwrap_or(0.0);

    let C = (gamma(2.0 - self.alpha as f64)
      * (self.lambda_plus.powf(self.alpha - 2.0) as f64
        + self.lambda_minus.powf(self.alpha - 2.0) as f64)) as f32;
    let C = C.powi(-1);

    let b_t = -C
      * gamma(1.0 - self.alpha as f64) as f32
      * (self.lambda_plus.powf(self.alpha - 1.0) - self.lambda_minus.powf(self.alpha - 1.0));

    let U = Array1::random(self.j, SimdUniform::new(0.0, 1.0));
    let E = Array1::random(self.j, SimdExp::new(1.0));
    let P = Poisson::<f32>::new(1.0, Some(self.j), None, None).sample_simd();
    let tau = Array1::random(self.j, SimdUniform::new(0.0, 1.0));

    for i in 1..self.n {
      let mut jump_component = 0.0;
      let t_1 = (i - 1) as f32 * dt;
      let t = i as f32 * dt;

      for j in 1..self.j {
        if tau[j] > t_1 && tau[j] <= t {
          let v_j = if rng.gen_bool(0.5) {
            self.lambda_plus
          } else {
            -self.lambda_minus
          };

          let divisor: f32 = 2.0 * C * t_max;
          let numerator: f32 = self.alpha * P[j];
          let term1 = (numerator / divisor).powf(-1.0 / self.alpha);
          let term2 = E[j] * U[j].powf(1.0 / self.alpha) / v_j.abs();
          let jump_size = term1.min(term2) * (v_j / v_j.abs());

          jump_component += jump_size;
        }
      }

      x[i] = x[i - 1] + jump_component + b_t * dt;
    }

    x
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Axis;

  use super::*;
  use crate::plot_1d;
  use crate::plot_nd;
  use crate::stochastic::N;

  #[test]
  fn cgmy_length_equals_n() {
    let cgmy = CGMY::new(5.0, 5.0, 0.7, N, 1000, Some(0.0), Some(1.0), None);
    assert_eq!(cgmy.sample().len(), N);
  }

  #[test]
  fn cgmy_starts_with_x0() {
    let cgmy = CGMY::new(5.0, 5.0, 0.7, N, 1000, Some(0.0), Some(1.0), None);
    assert_eq!(cgmy.sample()[0], 0.0);
  }

  #[test]
  fn cgmy_plot() {
    let cgmy = CGMY::new(25.46, 4.604, 0.52, 1000, 1024, Some(2.0), Some(1.0), None);
    plot_1d!(cgmy.sample(), "CGMY Process");
  }

  #[test]
  fn cgmy_plot_multi() {
    let cgmy = CGMY::new(25.46, 4.604, 0.52, N, 10000, Some(2.0), Some(1.0), Some(10));
    plot_nd!(cgmy.sample_par(), "CGMY Process");
  }
}
