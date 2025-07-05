use std::f64::consts::FRAC_1_PI;

use impl_new_derive::ImplNew;
use implied_vol::implied_black_volatility;
use nalgebra::Complex;
use num_complex::Complex64;
use quadrature::double_exponential;
use rand_distr::num_traits::ConstOne;

use crate::quant::{
  r#trait::{PricerExt, TimeExt},
  OptionType,
};

#[derive(ImplNew, Clone)]
pub struct HestonPricer {
  /// Stock price
  pub s: f64,
  /// Initial volatility
  pub v0: f64,
  /// Strike price
  pub k: f64,
  /// Risk-free rate
  pub r: f64,
  /// Dividend yield
  pub q: Option<f64>,
  /// Correlation between the stock price and its volatility
  pub rho: f64,
  /// Mean reversion rate
  pub kappa: f64,
  /// Long-run average volatility
  pub theta: f64,
  /// Volatility of volatility
  pub sigma: f64,
  /// Market price of volatility risk
  pub lambda: Option<f64>,
  /// Time to maturity
  pub tau: Option<f64>,
  /// Evaluation date
  pub eval: Option<chrono::NaiveDate>,
  /// Expiration date
  pub expiry: Option<chrono::NaiveDate>,
}

impl PricerExt for HestonPricer {
  /// Calculate the price of a European call option using the Heston model
  /// https://quant.stackexchange.com/a/18686
  fn calculate_call_put(&self) -> (f64, f64) {
    let tau = self.tau().unwrap_or(1.0);

    let call = self.s * (-self.q.unwrap_or(0.0) * tau).exp() * self.p(1, tau)
      - self.k * (-self.r * tau).exp() * self.p(2, tau);
    let put = call + self.k * (-self.r * tau).exp() - self.s * (-self.q.unwrap_or(0.0) * tau).exp();

    (call, put)
  }

  /// Derivatives
  fn derivatives(&self) -> Vec<f64> {
    let tau = self.tau().unwrap_or(1.0);

    vec![
      self.h1(tau),
      self.h2(tau),
      self.h3(tau),
      self.h4(tau),
      self.h5(tau),
    ]
  }

  fn implied_volatility(&self, c_price: f64, option_type: OptionType) -> f64 {
    implied_black_volatility(
      c_price,
      self.s,
      self.k,
      self.calculate_tau_in_days(),
      option_type == OptionType::Call,
    )
  }
}

impl TimeExt for HestonPricer {
  fn tau(&self) -> Option<f64> {
    self.tau
  }

  fn eval(&self) -> chrono::NaiveDate {
    self.eval.unwrap()
  }

  fn expiration(&self) -> chrono::NaiveDate {
    self.expiry.unwrap()
  }
}

impl HestonPricer {
  pub(self) fn u(&self, j: u8) -> f64 {
    match j {
      1 => 0.5,
      2 => -0.5,
      _ => panic!("Invalid j"),
    }
  }

  pub(self) fn b(&self, j: u8) -> f64 {
    match j {
      1 => self.kappa + self.lambda.unwrap_or(1.0) - self.rho * self.sigma,
      2 => self.kappa + self.lambda.unwrap_or(1.0),
      _ => panic!("Invalid j"),
    }
  }

  pub(self) fn d(&self, j: u8, phi: f64) -> Complex64 {
    ((self.b(j) - self.rho * self.sigma * phi * Complex64::i()).powi(2)
      - self.sigma.powi(2) * (2.0 * Complex64::i() * self.u(j) * phi - phi.powi(2)))
    .sqrt()
  }

  pub(self) fn g(&self, j: u8, phi: f64) -> Complex64 {
    (self.b(j) - self.rho * self.sigma * Complex64::i() * phi + self.d(j, phi))
      / (self.b(j) - self.rho * self.sigma * Complex64::i() * phi - self.d(j, phi))
  }

  pub(self) fn C(&self, j: u8, phi: f64, tau: f64) -> Complex64 {
    (self.r - self.q.unwrap_or(0.0)) * Complex64::i() * phi * tau
      + (self.kappa * self.theta / self.sigma.powi(2))
        * ((self.b(j) - self.rho * self.sigma * Complex64::i() * phi + self.d(j, phi)) * tau
          - 2.0
            * ((1.0 - self.g(j, phi) * (self.d(j, phi) * tau).exp()) / (1.0 - self.g(j, phi))).ln())
  }

  pub(self) fn D(&self, j: u8, phi: f64, tau: f64) -> Complex64 {
    ((self.b(j) - self.rho * self.sigma * Complex64::i() * phi + self.d(j, phi))
      / self.sigma.powi(2))
      * ((1.0 - (self.d(j, phi) * tau).exp())
        / (1.0 - self.g(j, phi) * (self.d(j, phi) * tau).exp()))
  }

  pub(self) fn f(&self, j: u8, phi: f64, tau: f64) -> Complex64 {
    (self.C(j, phi, tau) + self.D(j, phi, tau) * self.v0 + Complex64::i() * phi * self.s.ln()).exp()
  }

  pub(self) fn re(&self, j: u8, tau: f64) -> impl Fn(f64) -> f64 {
    let self_ = self.clone();
    move |phi: f64| -> f64 {
      (self_.f(j, phi, tau) * (-Complex64::i() * phi * self_.k.ln()).exp() / (Complex64::i() * phi))
        .re
    }
  }

  pub(self) fn p(&self, j: u8, tau: f64) -> f64 {
    0.5 + FRAC_1_PI * double_exponential::integrate(self.re(j, tau), 0.00001, 50.0, 10e-6).integral
  }

  /// Partial derivative of the C function with respect to parameters
  /// https://www.sciencedirect.com/science/article/abs/pii/S0377221717304460

  /// Partial derivative of the C function with respect to the v0 parameter
  pub(crate) fn h1(&self, tau: f64) -> f64 {
    -(self.A(tau) / self.v0).re
  }

  /// Partial derivative of the C function with respect to the theta parameter
  pub(crate) fn h2(&self, tau: f64) -> f64 {
    ((2.0 * self.kappa / self.sigma.powi(2)) * self.D_(tau)
      - self.kappa * self.rho * tau * Complex64::i() * self.u(1) / self.sigma)
      .re
  }

  /// Partial derivative of the C function with respect to the rho parameter
  pub(crate) fn h3(&self, tau: f64) -> f64 {
    (-self.dA_drho(tau)
      + ((2.0 * self.kappa * self.theta) / (self.sigma.powi(2) * self.d_()))
        * (self.dd__drho() - (self.d_() / self.A2(tau)) * self.dA2_drho(tau))
      - (self.kappa * self.theta * tau * Complex64::i() * self.u(1)) / self.sigma)
      .re
  }

  /// Partial derivative of the C function with respect to the kappa parameter
  pub(crate) fn h4(&self, tau: f64) -> f64 {
    ((1.0 / (self.sigma * Complex64::i() * self.u(1))) * self.dA_drho(tau)
      + ((2.0 * self.theta) / self.sigma.powi(2)) * self.D_(tau)
      + ((2.0 * self.kappa * self.theta) / (self.sigma.powi(2) * self.B(tau)))
        * self.dB_dkappa(tau)
      - (self.theta * self.rho * tau * Complex64::i() * self.u(1)) / self.sigma)
      .re
  }

  /// Partial derivative of the C function with respect to the sigma parameter
  pub(crate) fn h5(&self, tau: f64) -> f64 {
    (-self.dA_dsigma(tau) - ((4.0 * self.kappa * self.theta) / self.sigma.powi(3)) * self.D_(tau)
      + ((2.0 * self.kappa * self.theta) / (self.sigma.powi(2) * self.d_()))
        * (self.dd__dsigma() - (self.d_() / self.A2(tau)) * self.dA2_dsigma(tau))
      + (self.kappa * self.theta * self.rho * tau * Complex64::i() * self.u(1))
        / self.sigma.powi(2))
    .re
  }

  // helpers derivates
  pub(self) fn dxi_drho(&self) -> Complex64 {
    -self.sigma * Complex64::i() * self.u(1)
  }

  pub(self) fn dd__drho(&self) -> Complex64 {
    -(self.xi() * self.sigma * Complex64::i() * self.u(1)) / self.d_()
  }

  pub(self) fn dA1_drho(&self, tau: f64) -> Complex64 {
    -((Complex64::i()
      * self.u(1)
      * (self.u(1).powi(2) + Complex64::i() * self.u(1))
      * tau
      * self.xi()
      * self.sigma)
      / (2.0 * self.d_())
      * (self.d_() * tau / 2.0).cosh())
  }

  pub(self) fn dA2_drho(&self, tau: f64) -> Complex64 {
    ((self.dxi_drho() * (2.0 + self.xi() * tau)) / (2.0 * self.d_() * self.v0))
      * (self.xi() * (self.d_() * tau / 2.0).cosh() + self.d_() * (self.d_() * tau / 2.0).sinh())
  }

  pub(self) fn dA_drho(&self, tau: f64) -> Complex64 {
    (1.0 / self.A2(tau)) * self.dA1_drho(tau) - (self.A(tau) / self.A2(tau)) * self.dA2_drho(tau)
  }

  pub(self) fn dd__dsigma(&self) -> Complex64 {
    (self.rho / self.sigma - 1.0 / self.xi()) * self.dd__drho()
      + (self.sigma * self.u(1).powi(2) / self.d_())
  }

  pub(self) fn dA1_dsigma(&self, tau: f64) -> Complex64 {
    (((self.u(1).powi(2) + Complex64::i() * self.u(1)) * tau) / 2.0)
      * self.dd__dsigma()
      * (self.d_() * tau / 2.0).cosh()
  }

  pub(self) fn dA2_dsigma(&self, tau: f64) -> Complex64 {
    (self.rho / self.sigma) * self.dA2_drho(tau)
      - ((2.0 + tau * self.xi()) / (self.v0 * tau * self.xi() * Complex64::i() * self.u(1)))
        * self.dA1_drho(tau)
      + (self.sigma * tau * self.A1(tau)) / (2.0 * self.v0)
  }

  pub(self) fn dA_dsigma(&self, tau: f64) -> Complex64 {
    (1.0 / self.A2(tau)) * self.dA1_dsigma(tau)
      - (self.A(tau) / self.A2(tau)) * self.dA2_dsigma(tau)
  }

  pub(self) fn dB_drho(&self, tau: f64) -> Complex64 {
    ((self.kappa * tau / 2.0).exp() / self.v0)
      * ((1.0 / self.A2(tau)) * self.dd__drho()
        - (self.d_() / self.A2(tau).powi(2)) * self.dA2_drho(tau))
  }

  pub(self) fn dB_dkappa(&self, tau: f64) -> Complex64 {
    (Complex64::i() / (self.sigma * self.u(1))) * self.dB_drho(tau) + (self.B(tau) * tau) / 2.0
  }

  pub(self) fn xi(&self) -> Complex64 {
    self.kappa - self.sigma * self.rho * Complex64::i() * self.u(1)
  }

  pub(self) fn d_(&self) -> Complex64 {
    (self.xi().powi(2) + self.sigma.powi(2) * (self.u(1).powi(2) + Complex64::i() * self.u(1)))
      .sqrt()
  }

  pub(self) fn A1(&self, tau: f64) -> Complex64 {
    (self.u(1).powi(2) + Complex64::i() * self.u(1)) * (self.d_() * tau / 2.0).sinh()
  }

  pub(self) fn A2(&self, tau: f64) -> Complex64 {
    (self.d_() / self.v0) * (self.d_() * tau / 2.0).cosh()
      + (self.xi() / self.v0) * (self.d_() * tau / 2.0).sinh()
  }

  pub(self) fn A(&self, tau: f64) -> Complex64 {
    self.A1(tau) / self.A2(tau)
  }

  pub(self) fn B(&self, tau: f64) -> Complex64 {
    (self.d_() * (self.kappa * tau / 2.0).exp()) / (self.v0 * self.A2(tau))
  }

  pub(self) fn D_(&self, tau: f64) -> Complex64 {
    self.B(tau).ln()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn heston_single_price() {
    let heston = HestonPricer::new(
      100.0,
      0.05,
      90.0,
      0.03,
      Some(0.02),
      -0.8,
      5.0,
      0.05,
      0.5,
      Some(0.0),
      Some(0.5),
      None,
      None,
    );

    let (call, put) = heston.calculate_call_put();
    println!("Call Price: {}, Put Price: {}", call, put);
  }

  #[test]
  fn heston_implied_volatility() {
    let heston = HestonPricer::new(
      100.0,
      0.05,
      90.0,
      0.03,
      Some(0.02),
      -0.8,
      5.0,
      0.05,
      0.5,
      Some(0.0),
      Some(1.0),
      None,
      None,
    );

    let (call, ..) = heston.calculate_call_put();
    let iv = heston.implied_volatility(call, OptionType::Call);
    println!("Implied Volatility: {}", iv);
  }
}
