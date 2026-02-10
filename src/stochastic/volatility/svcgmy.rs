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
use crate::stats::non_central_chi_squared;
use crate::stochastic::process::poisson::Poisson;
use crate::stochastic::Float;
use crate::stochastic::Process;

/// CGMY Stochastic Volatility process
///
/// https://www.econstor.eu/bitstream/10419/239493/1/175133161X.pdf
pub struct SVCGMY<T: Float> {
  /// Positive jump rate lambda_plus (corresponds to G)
  pub lambda_plus: T, // G
  /// Negative jump rate lambda_minus (corresponds to M)
  pub lambda_minus: T, // M
  /// Jump activity parameter alpha (corresponds to Y), with 0 < alpha < 2
  pub alpha: T,
  /// Mean reversion rate
  pub kappa: T,
  /// Long-term volatility
  pub eta: T,
  /// Volatility of volatility
  pub zeta: T,
  ///
  pub rho: T,
  /// Number of time steps
  pub n: usize,
  /// Jumps
  pub j: usize,
  /// Initial value
  pub x0: Option<T>,
  /// Initial value
  pub v0: Option<T>,
  /// Total time horizon
  pub t: Option<T>,
}

impl<T: Float> SVCGMY<T> {
  pub fn new(
    lambda_plus: T,
    lambda_minus: T,
    alpha: T,
    kappa: T,
    eta: T,
    zeta: T,
    rho: T,
    n: usize,
    j: usize,
    x0: Option<T>,
    v0: Option<T>,
    t: Option<T>,
  ) -> Self {
    Self {
      lambda_plus,
      lambda_minus,
      alpha,
      kappa,
      eta,
      zeta,
      rho,
      n,
      j,
      x0,
      v0,
      t,
    }
  }
}

impl<T: Float> Process<T> for SVCGMY<T> {
  type Output = Array1<T>;
  type Noise = Self;

  fn sample(&self) -> Self::Output {
    let mut rng = rand::rng();

    let t_max = self.t.unwrap_or(T::one());
    let dt = t_max / T::from_usize(self.n - 1);

    let mut x = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);
    let mut y = Array1::<T>::zeros(self.n);

    x[0] = self.x0.unwrap_or(T::zero());
    v[0] = self.v0.unwrap_or(T::zero());

    let C = T::one()
      / (gamma(2.0 - self.alpha)
        * (self.lambda_plus.powf(self.alpha - T::from_usize(2))
          + self.lambda_minus.powf(self.alpha - T::from_usize(2))));
    let c =
      (T::from_usize(2) * self.kappa) / ((T::one() - (-self.kappa * dt).exp()) * self.zeta.powi(2));
    let df = T::from_usize(4) * self.kappa * self.eta / self.zeta.powi(2);

    for i in 1..self.n {
      let ncp = T::from_usize(2) * c * v[i - 1] * (-self.kappa * dt).exp();
      let xi = non_central_chi_squared::sample(df, ncp, &mut rng);
      v[i] = xi / (T::from_usize(2) * c);
    }

    #[cfg(not(feature = "simd"))]
    let uniform = Uniform::new(T::zero(), T::one()).unwrap();
    #[cfg(not(feature = "simd"))]
    let exp = Exp::new(T::one()).unwrap();

    #[cfg(feature = "simd")]
    let uniform = SimdUniform::new(T::zero(), T::one());
    #[cfg(feature = "simd")]
    let exp = SimdExp::new(T::one());

    let U = Array1::<T>::random(self.j, uniform);
    let E = Array1::<T>::random(self.j, exp);
    let P = Poisson::new(T::one(), Some(self.j), None);

    #[cfg(not(feature = "simd"))]
    let P = P.sample();

    #[cfg(feature = "simd")]
    let P = P.sample_simd();

    let tau = Array1::<f64>::random(self.j, uniform) * t_max;

    let mut c_tau = Array1::<T>::zeros(self.j);
    for (idx, tau_j) in tau.iter().enumerate() {
      let k = ((tau_j / dt).ceil() as usize).min(self.n - 1);
      let v_k = if k == 0 { v[0] } else { v[k - 1] };
      c_tau[idx] = C * v_k;
    }

    for i in 1..self.n {
      let numerator = v[i - 1]
        * (self.lambda_plus.powf(self.alpha - T::one())
          - self.lambda_minus.powf(self.alpha - T::one()));
      let denominator = (T::one() - self.alpha)
        * (self.lambda_plus.powf(self.alpha - T::from_usize(2))
          + self.lambda_minus.powf(self.alpha - T::from_usize(2)));
      let b = -numerator / denominator;

      let mut jump_component = T::zero();

      let t_1 = T::from_usize(i - 1) * dt;
      let t = T::from_usize(i) * dt;

      for j in 0..self.j {
        if tau[j] > t_1 && tau[j] <= t {
          let v_j = if rng.random_bool(0.5) {
            self.lambda_plus
          } else {
            -self.lambda_minus
          };

          let numerator = self.alpha * P[j];
          let denominator = T::from_usize(2) * c_tau[j] * t_max;
          let term1 = (numerator / denominator).powf(-T::one() / self.alpha);
          let term2 = E[j] * U[j].powf(T::one() / self.alpha) / v_j.abs();
          let min_term = term1.min(term2);
          let jump_size = min_term * (v_j / v_j.abs());
          jump_component += jump_size;
        }
      }

      y[i] = y[i - 1] + jump_component + b * dt;
    }

    for i in 1..self.n {
      x[i] = y[i] + self.rho * v[i];
    }

    x
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    self.sample()
  }

  fn euler_maruyama(
    &self,
    _noise_fn: impl Fn(&Self::Noise) -> <Self::Noise as Process<T>>::Output,
  ) -> Self::Output {
    unimplemented!()
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
  fn svcgmy_length_equals_n() {
    let svcgmy = SVCGMY::new(
      25.46,
      4.604,
      0.52,
      1.003,
      0.0711,
      0.3443,
      -2.0280,
      N,
      1024,
      None,
      Some(0.0064),
      Some(1.0),
    );
    assert_eq!(svcgmy.sample().len(), N);
  }

  #[test]
  fn svcgmy_starts_with_x0() {
    let svcgmy = SVCGMY::new(
      25.46,
      4.604,
      0.52,
      1.003,
      0.0711,
      0.3443,
      -2.0280,
      N,
      1024,
      None,
      Some(0.0064),
      Some(1.0),
    );
    assert_eq!(svcgmy.sample()[0], 0.0);
  }

  #[test]
  fn svcgmy_plot() {
    let svcgmy = SVCGMY::new(
      25.46,
      4.604,
      0.52,
      1.003,
      0.0711,
      0.3443,
      -2.0280,
      1000,
      1024,
      Some(-0.25),
      Some(0.0064),
      Some(1.0),
    );
    plot_1d!(svcgmy.sample(), "SVCGMY Process");
  }

  #[test]
  fn svcgmy_plot_multi() {
    let svcgmy = SVCGMY::new(
      25.46,
      4.604,
      0.52,
      1.003,
      0.0711,
      0.3443,
      -2.0280,
      1000,
      1024,
      Some(-0.25),
      Some(0.0064),
      Some(1.0),
    );
    plot_nd!(svcgmy.sample_par(10), "SVCGMY Process");
  }
}
