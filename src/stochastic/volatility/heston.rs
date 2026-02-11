use ndarray::Array1;

use super::HestonPow;
use crate::stochastic::noise::cgns::CGNS;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct Heston<T: Float> {
  /// Initial stock price
  pub s0: Option<T>,
  /// Initial volatility
  pub v0: Option<T>,
  /// Mean reversion rate
  pub kappa: T,
  /// Long-run average volatility
  pub theta: T,
  /// Volatility of volatility
  pub sigma: T,
  /// Correlation between the stock price and its volatility
  pub rho: T,
  /// Drift of the stock price
  pub mu: T,
  /// Number of time steps
  pub n: usize,
  /// Time to maturity
  pub t: Option<T>,
  /// Power of the variance
  /// If 0.5 then it is the original Heston model
  /// If 1.5 then it is the 3/2 model
  pub pow: HestonPow,
  /// Use the symmetric method for the variance to avoid negative values
  pub use_sym: Option<bool>,
  /// Noise generator
  cgns: CGNS<T>,
}

impl<T: Float> Heston<T> {
  pub fn new(
    s0: Option<T>,
    v0: Option<T>,
    kappa: T,
    theta: T,
    sigma: T,
    rho: T,
    mu: T,
    n: usize,
    t: Option<T>,
    pow: HestonPow,
    use_sym: Option<bool>,
  ) -> Self {
    Self {
      s0,
      v0,
      kappa,
      theta,
      sigma,
      rho,
      mu,
      n,
      t,
      pow,
      use_sym,
      cgns: CGNS::new(rho, n - 1, t),
    }
  }
}

impl<T: Float> Process<T> for Heston<T> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let dt = self.cgns.dt();
    let [cgn1, cgn2] = &self.cgns.sample();

    let mut s = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);

    s[0] = self.s0.unwrap_or(T::zero());
    v[0] = self.v0.unwrap_or(T::zero());

    for i in 1..self.n {
      s[i] = s[i - 1] + self.mu * s[i - 1] * dt + s[i - 1] * v[i - 1].sqrt() * cgn1[i - 1];

      let dv = self.kappa * (self.theta - v[i - 1]) * dt
        + self.sigma
          * v[i - 1].powf(match self.pow {
            HestonPow::Sqrt => T::from_f64_fast(0.5),
            HestonPow::ThreeHalves => T::from_f64_fast(1.5),
          })
          * cgn2[i - 1];

      v[i] = match self.use_sym.unwrap_or(false) {
        true => (v[i - 1] + dv).abs(),
        false => (v[i - 1] + dv).max(T::zero()),
      }
    }

    [s, v]
  }
}

impl<T: Float> Heston<T> {
  /// Malliavin derivative of the volatility
  ///
  /// The Malliavin derivative of the Heston model is given by
  /// D_r v_t = \sigma v_t^{1/2} / 2 * exp(-(\kappa \theta / 2 - \sigma^2 / 8) / v_t * dt)
  ///
  /// The Malliavin derivative of the 3/2 Heston model is given by
  /// D_r v_t = \sigma v_t^{3/2} / 2 * exp(-(\kappa \theta / 2 + 3 \sigma^2 / 8) * v_t * dt)
  pub fn malliavin_of_vol(&self) -> [Array1<T>; 3] {
    let [s, v] = self.sample();
    let dt = self.t.unwrap_or(T::zero()) / T::from_usize_(self.n - 1);

    let mut det_term = Array1::zeros(self.n);
    let mut malliavin = Array1::zeros(self.n);
    let f2 = T::from_usize_(2);

    for i in 0..self.n {
      match self.pow {
        HestonPow::Sqrt => {
          det_term[i] = ((-(self.kappa * self.theta / f2
            - self.sigma.powi(2) / T::from_usize_(8))
            * (T::one() / *v.last().unwrap())
            - self.kappa / f2)
            * (T::from_usize_(self.n - i) * dt))
            .exp();
          malliavin[i] = (self.sigma * v.last().unwrap().sqrt() / f2) * det_term[i];
        }
        HestonPow::ThreeHalves => {
          det_term[i] = ((-(self.kappa * self.theta / f2
            + T::from_usize_(3) * self.sigma.powi(2) / T::from_usize_(8))
            * *v.last().unwrap()
            - (self.kappa * self.theta) / f2)
            * (T::from_usize_(self.n - i) * dt))
            .exp();
          malliavin[i] =
            (self.sigma * v.last().unwrap().powf(T::from_f64_fast(1.5)) / f2) * det_term[i];
        }
      };
    }

    [s, v, malliavin]
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::plot_2d;
  use crate::stochastic::N;
  use crate::stochastic::S0;
  use crate::stochastic::X0;

  #[test]
  fn heston_malliavin() {
    let heston = Heston::new(
      Some(S0),
      Some(X0),
      0.5,
      1.0,
      1.0,
      1.0,
      1.0,
      N,
      Some(1.0),
      HestonPow::Sqrt,
      None,
    );
    let process = heston.sample();
    let malliavin = heston.malliavin_of_vol();
    plot_2d!(
      process[1],
      "Heston volatility process",
      malliavin[1],
      "Malliavin derivative of the Heston volatility process"
    );
  }
}
