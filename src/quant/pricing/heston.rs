//! # Heston
//!
//! $$
//! \begin{aligned}dS_t&=\mu S_tdt+\sqrt{v_t}S_tdW_t^S\\dv_t&=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dW_t^v,\ d\langle W^S,W^v\rangle_t=\rho dt\end{aligned}
//! $$
//!
use std::f64::consts::FRAC_1_PI;

use implied_vol::implied_black_volatility;
use num_complex::Complex64;
use quadrature::double_exponential;

use crate::quant::OptionType;
use crate::traits::PricerExt;
use crate::traits::TimeExt;

#[derive(Clone)]
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

impl HestonPricer {
  pub fn new(
    s: f64,
    v0: f64,
    k: f64,
    r: f64,
    q: Option<f64>,
    rho: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    lambda: Option<f64>,
    tau: Option<f64>,
    eval: Option<chrono::NaiveDate>,
    expiry: Option<chrono::NaiveDate>,
  ) -> Self {
    Self {
      s,
      v0,
      k,
      r,
      q,
      rho,
      kappa,
      theta,
      sigma,
      lambda,
      tau,
      eval,
      expiry,
    }
  }

  pub fn builder(
    s: f64,
    v0: f64,
    k: f64,
    r: f64,
    rho: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
  ) -> HestonPricerBuilder {
    HestonPricerBuilder {
      s,
      v0,
      k,
      r,
      q: None,
      rho,
      kappa,
      theta,
      sigma,
      lambda: None,
      tau: None,
      eval: None,
      expiry: None,
    }
  }
}

pub struct HestonPricerBuilder {
  s: f64,
  v0: f64,
  k: f64,
  r: f64,
  q: Option<f64>,
  rho: f64,
  kappa: f64,
  theta: f64,
  sigma: f64,
  lambda: Option<f64>,
  tau: Option<f64>,
  eval: Option<chrono::NaiveDate>,
  expiry: Option<chrono::NaiveDate>,
}

impl HestonPricerBuilder {
  pub fn q(mut self, q: f64) -> Self {
    self.q = Some(q);
    self
  }
  pub fn lambda(mut self, lambda: f64) -> Self {
    self.lambda = Some(lambda);
    self
  }
  pub fn tau(mut self, tau: f64) -> Self {
    self.tau = Some(tau);
    self
  }
  pub fn eval(mut self, eval: chrono::NaiveDate) -> Self {
    self.eval = Some(eval);
    self
  }
  pub fn expiry(mut self, expiry: chrono::NaiveDate) -> Self {
    self.expiry = Some(expiry);
    self
  }
  pub fn build(self) -> HestonPricer {
    HestonPricer {
      s: self.s,
      v0: self.v0,
      k: self.k,
      r: self.r,
      q: self.q,
      rho: self.rho,
      kappa: self.kappa,
      theta: self.theta,
      sigma: self.sigma,
      lambda: self.lambda,
      tau: self.tau,
      eval: self.eval,
      expiry: self.expiry,
    }
  }
}

impl PricerExt for HestonPricer {
  fn calculate_call_put(&self) -> (f64, f64) {
    let tau = self.tau().unwrap_or(1.0);

    let call = self.s * (-self.q.unwrap_or(0.0) * tau).exp() * self.p(1, tau)
      - self.k * (-self.r * tau).exp() * self.p(2, tau);
    let put = call + self.k * (-self.r * tau).exp() - self.s * (-self.q.unwrap_or(0.0) * tau).exp();

    (call, put)
  }

  fn calculate_price(&self) -> f64 {
    self.calculate_call_put().0
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

  fn eval(&self) -> Option<chrono::NaiveDate> {
    self.eval
  }

  fn expiration(&self) -> Option<chrono::NaiveDate> {
    self.expiry
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
      1 => self.kappa + self.lambda.unwrap_or(0.0) - self.rho * self.sigma,
      2 => self.kappa + self.lambda.unwrap_or(0.0),
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
