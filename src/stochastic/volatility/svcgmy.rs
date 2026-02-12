use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand::Rng;
use scilib::math::basic::gamma;

use crate::distributions::exp::SimdExp;
use crate::distributions::uniform::SimdUniform;
use crate::stats::non_central_chi_squared;
use crate::stochastic::process::poisson::Poisson;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// CGMY Stochastic Volatility process
///
/// https://www.econstor.eu/bitstream/10419/239493/1/175133161X.pdf
pub struct SVCGMY<T: FloatExt> {
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

impl<T: FloatExt> SVCGMY<T> {
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

impl<T: FloatExt> ProcessExt<T> for SVCGMY<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut rng = rand::rng();

    let t_max = self.t.unwrap_or(T::one());
    let dt = t_max / T::from_usize_(self.n - 1);

    let mut x = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);
    let mut y = Array1::<T>::zeros(self.n);

    x[0] = self.x0.unwrap_or(T::zero());
    v[0] = self.v0.unwrap_or(T::zero());

    let f2 = T::from_usize_(2);

    let g = gamma(2.0 - self.alpha.to_f64().unwrap());
    let C = T::one()
      / (T::from_f64_fast(g)
        * (self.lambda_plus.powf(self.alpha - f2) + self.lambda_minus.powf(self.alpha - f2)));
    let c = (f2 * self.kappa) / ((T::one() - (-self.kappa * dt).exp()) * self.zeta.powi(2));
    let df = T::from_usize_(4) * self.kappa * self.eta / self.zeta.powi(2);

    for i in 1..self.n {
      let ncp = f2 * c * v[i - 1] * (-self.kappa * dt).exp();
      let xi = non_central_chi_squared::sample(df, ncp, &mut rng);
      v[i] = xi / (f2 * c);
    }

    let uniform = SimdUniform::new(T::zero(), T::one());
    let exp = SimdExp::new(T::one());

    let U = Array1::<T>::random(self.j, &uniform);
    let E = Array1::<T>::random(self.j, exp);
    let P = Poisson::new(T::one(), Some(self.j), None);
    let P = P.sample();
    let tau = Array1::<T>::random(self.j, &uniform) * t_max;

    let mut c_tau = Array1::<T>::zeros(self.j);
    for (idx, tau_j) in tau.iter().enumerate() {
      let k = ((*tau_j / dt).ceil()).min(T::from_usize_(self.n - 1));
      let v_k = if k == T::zero() {
        v[0]
      } else {
        v[k.to_usize().unwrap() - 1]
      };
      c_tau[idx] = C * v_k;
    }

    for i in 1..self.n {
      let numerator = v[i - 1]
        * (self.lambda_plus.powf(self.alpha - T::one())
          - self.lambda_minus.powf(self.alpha - T::one()));
      let denominator = (T::one() - self.alpha)
        * (self.lambda_plus.powf(self.alpha - f2) + self.lambda_minus.powf(self.alpha - f2));
      let b = -numerator / denominator;

      let mut jump_component = T::zero();

      let t_1 = T::from_usize_(i - 1) * dt;
      let t = T::from_usize_(i) * dt;

      for j in 0..self.j {
        if tau[j] > t_1 && tau[j] <= t {
          let v_j = if rng.random_bool(0.5) {
            self.lambda_plus
          } else {
            -self.lambda_minus
          };

          let numerator = self.alpha * P[j];
          let denominator = f2 * c_tau[j] * t_max;
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
}
