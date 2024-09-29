use std::{cell::RefCell, f64::consts::FRAC_1_PI};

use either::Either;
use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DVector;
use num_complex::Complex64;
use quadrature::double_exponential;

use crate::{volatility::Calibrator, yahoo::Yahoo};

use super::Pricer;

#[derive(Default, Clone)]
pub struct HestonPricer {
  /// Initial stock price
  pub s0: f64,
  /// Initial volatility
  pub v0: f64,
  /// Strike price
  pub k: f64,
  /// Risk-free rate
  pub r: f64,
  /// Dividend yield
  pub q: f64,
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
  pub tau: Option<Either<f64, Vec<f64>>>,
  /// Evaluation date
  pub eval: Option<Either<chrono::NaiveDate, Vec<chrono::NaiveDate>>>,
  /// Expiration date
  pub expiry: Option<Either<chrono::NaiveDate, Vec<chrono::NaiveDate>>>,
  /// Prices of European call and put options
  pub(crate) prices: Option<Either<(f64, f64), Vec<(f64, f64)>>>,
  /// Partial derivative of the C function with respect to the parameters
  pub(crate) derivates: Option<Either<Vec<f64>, Vec<Vec<f64>>>>,
}

impl Pricer for HestonPricer {
  /// Calculate the price of a European call option using the Heston model
  /// https://quant.stackexchange.com/a/18686
  fn calculate_price(&mut self) -> Either<(f64, f64), Vec<(f64, f64)>> {
    if self.tau.is_none() && self.eval.is_none() && self.expiry.is_none() {
      panic!("At least 2 of tau, eval, and expiry must be provided");
    }

    let tau = self.tau.as_ref().unwrap();
    if tau.as_ref().right().is_none() {
      let tau = tau.as_ref().left().unwrap();

      let call = self.s0 * (-self.q * tau).exp() * self.p(1, *tau)
        - self.k * (-self.r * tau).exp() * self.p(2, *tau);
      let put = call + self.k * (-self.r * tau).exp() - self.s0 * (-self.q * tau).exp();

      self.prices = Some(Either::Left((call, put)));
      self.derivates = Some(Either::Left(self.derivates(*tau)));
      Either::Left((call, put))
    } else {
      let tau = tau.as_ref().right().unwrap();
      let mut prices = Vec::with_capacity(tau.len());
      let mut derivatives = Vec::with_capacity(tau.len());

      for tau in tau.iter() {
        let call = self.s0 * (-self.q * tau).exp() * self.p(1, *tau)
          - self.k * (-self.r * tau).exp() * self.p(2, *tau);
        let put = call + self.k * (-self.r * tau).exp() - self.s0 * (-self.q * tau).exp();

        prices.push((call, put));
        derivatives.push(self.derivates(*tau));
      }

      self.prices = Some(Either::Right(prices.clone()));
      self.derivates = Some(Either::Right(derivatives));
      Either::Right(prices)
    }
  }

  /// Update the parameters from the calibration
  fn update_params(&mut self, params: DVector<f64>) {
    self.v0 = params[0];
    self.theta = params[1];
    self.rho = params[2];
    self.kappa = params[3];
    self.sigma = params[4];
  }

  /// Prices.
  fn prices(&self) -> Option<Either<(f64, f64), Vec<(f64, f64)>>> {
    self.prices.clone()
  }

  /// Derivatives.
  fn derivates(&self) -> Option<Either<Vec<f64>, Vec<Vec<f64>>>> {
    self.derivates.clone()
  }
}

impl HestonPricer {
  /// Create a new Heston model
  #[must_use]
  pub fn new(params: &Self) -> Self {
    Self {
      s0: params.s0,
      v0: params.v0,
      k: params.k,
      r: params.r,
      q: params.q,
      rho: params.rho,
      kappa: params.kappa,
      theta: params.theta,
      sigma: params.sigma,
      lambda: Some(params.lambda.unwrap_or(0.0)),
      tau: params.tau.clone(),
      eval: params.eval.clone(),
      expiry: params.expiry.clone(),
      prices: None,
      derivates: None,
    }
  }

  pub(self) fn u(&self, j: u8) -> f64 {
    match j {
      1 => 0.5,
      2 => -0.5,
      _ => panic!("Invalid j"),
    }
  }

  pub(self) fn b(&self, j: u8) -> f64 {
    match j {
      1 => self.kappa + self.lambda.unwrap() - self.rho * self.sigma,
      2 => self.kappa + self.lambda.unwrap(),
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
    (self.r - self.q) * Complex64::i() * phi * tau
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
    (self.C(j, phi, tau) + self.D(j, phi, tau) * self.v0 + Complex64::i() * phi * self.s0.ln())
      .exp()
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
  pub(crate) fn dC_dv0(&self, tau: f64) -> f64 {
    (-self.A(tau) / self.v0).re
  }

  /// Partial derivative of the C function with respect to the theta parameter
  pub(crate) fn dC_dtheta(&self, tau: f64) -> f64 {
    ((2.0 * self.kappa / self.sigma.powi(2)) * self.D_(tau)
      - self.kappa * self.rho * tau * Complex64::i() * self.u(1) / self.sigma)
      .re
  }

  /// Partial derivative of the C function with respect to the rho parameter
  pub(crate) fn dC_drho(&self, tau: f64) -> f64 {
    (-self.kappa * self.theta * tau * Complex64::i() * self.u(1) / self.sigma).re
  }

  /// Partial derivative of the C function with respect to the kappa parameter
  pub(crate) fn dC_dkappa(&self, tau: f64) -> f64 {
    (2.0 * self.theta * self.D_(tau) / self.sigma.powi(2)
      + ((2.0 * self.kappa * self.theta) / self.sigma.powi(2) * self.B(tau)) * self.dB_dkappa(tau)
      - (self.theta * self.rho * tau * Complex64::i() * self.u(1) / self.sigma))
      .re
  }

  /// Partial derivative of the C function with respect to the sigma parameter
  pub(crate) fn dC_dsigma(&self, tau: f64) -> f64 {
    ((-4.0 * self.kappa * self.theta / self.sigma.powi(3)) * self.D_(tau)
      + ((2.0 * self.kappa * self.theta) / (self.sigma.powi(2) * self.d_())) * self.dd_dsigma()
      + self.kappa * self.theta * self.rho * tau * Complex64::i() * self.u(1) / self.sigma.powi(2))
    .re
  }

  pub(self) fn xi(&self) -> Complex64 {
    self.kappa - self.sigma * self.rho * Complex64::i() * self.u(1)
  }

  pub(self) fn d_(&self) -> Complex64 {
    (self.xi().powi(2) + self.sigma.powi(2) * (self.u(1).powi(2) + Complex64::i() * self.u(1)))
      .sqrt()
  }

  pub(self) fn dd_dsigma(&self) -> Complex64 {
    (self.sigma * (self.u(1) + Complex64::i() * self.u(1))) / self.d_()
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

  pub(self) fn D_(&self, tau: f64) -> Complex64 {
    (self.d_() / self.v0).ln() + (self.kappa - self.d_() / 2.0) * tau
      - (((self.d_() + self.xi()) / (2.0 * self.v0))
        + ((self.d_() - self.xi()) / (2.0 * self.v0)) * (-self.d_() * tau).exp())
      .ln()
  }

  pub(self) fn B(&self, tau: f64) -> Complex64 {
    (self.d_() * (self.kappa * tau / 2.0).exp()) / (self.v0 * self.A2(tau))
  }

  pub(self) fn dB_dkappa(&self, tau: f64) -> Complex64 {
    (self.d_() * tau * (self.kappa * tau / 2.0).exp()) / (2.0 * self.v0 * self.A2(tau))
  }

  pub(crate) fn derivates(&self, tau: f64) -> Vec<f64> {
    vec![
      self.dC_dv0(tau),
      self.dC_dtheta(tau),
      self.dC_drho(tau),
      self.dC_dkappa(tau),
      self.dC_dsigma(tau),
    ]
  }
}

/// Heston calibrator
pub struct HestonCalibrator<'a> {
  /// Yahoo struct
  pub yahoo: Option<Yahoo<'a>>,
  /// Implied volatility vector
  pub v: Option<Vec<f64>>,
  /// Underlying asset prices vector
  pub s: Option<Vec<f64>>,
  /// Option prices vector from the market
  pub c: Option<Vec<f64>>,
  /// Heston pricer
  pricer: HestonPricer,
  /// Initial guess for the calibration from the NMLE method
  pub initial_guess: Option<DVector<f64>>,
}

impl<'a> HestonCalibrator<'a> {
  #[must_use]
  pub fn new(
    pricer: HestonPricer,
    yahoo: Option<Yahoo<'a>>,
    v: Option<Vec<f64>>,
    s: Option<Vec<f64>>,
    c: Option<Vec<f64>>,
  ) -> Self {
    Self {
      pricer,
      yahoo,
      v,
      s,
      c,
      initial_guess: None,
    }
  }

  pub fn calibrate(&mut self) {
    if (self.c.is_none() && self.v.is_none()) || self.yahoo.is_none() {
      panic!("Yahoo struct or s and v must be provided");
    }
    self.initial_guess();

    // Overwrite the pricer with the initial guess
    self.pricer.v0 = self.initial_guess.as_ref().unwrap()[0];
    self.pricer.theta = self.initial_guess.as_ref().unwrap()[1];
    self.pricer.rho = self.initial_guess.as_ref().unwrap()[2];
    self.pricer.kappa = self.initial_guess.as_ref().unwrap()[3];
    self.pricer.sigma = self.initial_guess.as_ref().unwrap()[4];

    // Print the initial guess
    println!(
      "Initial guess: v0: {}, theta: {}, rho: {}, kappa: {}, sigma {}",
      self.pricer.v0, self.pricer.theta, self.pricer.rho, self.pricer.kappa, self.pricer.sigma
    );

    // Calculate the price of the options using the initial guess
    self.pricer.calculate_price();

    // Calibrate the Heston model
    let pricer = RefCell::new(self.pricer.clone());
    let (result, ..) = LevenbergMarquardt::new().minimize(Calibrator::new(
      self.initial_guess.as_ref().unwrap().clone(),
      Some(DVector::from_vec(self.c.as_ref().unwrap().clone())),
      &pricer,
    ));

    // Overwrite the pricer with the calibrated parameters
    self.pricer.v0 = result.params[0];
    self.pricer.theta = result.params[1];
    self.pricer.rho = result.params[2];
    self.pricer.kappa = result.params[3];
    self.pricer.sigma = result.params[4];

    // Print the calibrated parameters
    println!(
      "Calibrated parameters: v0: {}, theta: {}, rho: {}, kappa: {}, sigma {}",
      self.pricer.v0, self.pricer.theta, self.pricer.rho, self.pricer.kappa, self.pricer.sigma
    );

    // Calculate the price of the options using the calibrated parameters
    self.pricer.calculate_price();
  }

  /// Initial guess for the calibration
  /// http://scis.scichina.com/en/2018/042202.pdf
  ///
  /// Using NMLE (Normal Maximum Likelihood Estimation) method
  fn initial_guess(&mut self) {
    if self.v.is_none() && self.c.is_none() {
      let yahoo = self.yahoo.as_mut().unwrap();
      // get options chain from yahoo
      yahoo.get_options_chain();
      yahoo.get_price_history();
      let options = yahoo.options.as_ref().unwrap();
      // get impl_volatities col from options
      let impl_vol = options.select(["impl_volatility"]).unwrap();
      // convert to vec
      let impl_vol = impl_vol
        .select_at_idx(0)
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect::<Vec<f64>>();
      self.v = Some(impl_vol.clone());

      let c = options.select(["last_price"]).unwrap();
      let c = c
        .select_at_idx(0)
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect::<Vec<f64>>();
      self.c = Some(c);

      let s = yahoo
        .price_history
        .as_ref()
        .unwrap()
        .select(["close"])
        .unwrap();
      let s = s
        .select_at_idx(0)
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect::<Vec<f64>>();
      self.s = Some(s);
    };

    let impl_vol = self.v.as_ref().unwrap();
    let n = impl_vol.len();
    let delta = 1.0 / n as f64;
    let mut sum = [0.0; 6];

    for i in 1..n {
      // sum of sqrt(V_i * V_{i-1})
      sum[0] += (impl_vol[i] * impl_vol[i - 1]).sqrt();

      // sum of sqrt(V_i / V_{i-1})
      sum[1] += (impl_vol[i] / impl_vol[i - 1]).sqrt();

      // sum of V_i
      sum[2] += impl_vol[i];

      // sum of V_{i-1}
      sum[3] += impl_vol[i - 1];

      // sum of sqrt(V_i)
      sum[4] += impl_vol[i].sqrt();

      // sum of sqrt(V_{i-1})
      sum[5] += impl_vol[i - 1].sqrt();
    }

    let P_hat = ((1.0 / n as f64) * sum[0] - (1.0 / n as f64).powi(2) * sum[1] * sum[3])
      / ((delta / 2.0) - (delta / 2.0) * (1.0 / n as f64).powi(2) * (1.0 / sum[3]) * sum[3]);

    let kappa_hat = (2.0 / delta)
      * (1.0 + (P_hat * delta / 2.0) * (1.0 / n as f64) * (1.0 / sum[3])
        - (1.0 / n as f64) * sum[1]);

    let sigma_hat = ((4.0 / delta)
      * (1.0 / n as f64)
      * (sum[4] - sum[5] - (delta / (2.0 * sum[5])) * (P_hat - kappa_hat * sum[3])).powi(2))
    .sqrt();

    let theta_hat = (P_hat + 0.25 * sigma_hat.powi(2)) / kappa_hat;

    let s = self.s.as_ref().unwrap();
    let mut sum_dw1dw2 = 0.0;
    for i in 1..n {
      let dw1_i = (s[i].ln() - s[i - 1].ln() - (self.pricer.r - 0.5 * impl_vol[i - 1]) * delta)
        / impl_vol[i - 1].sqrt();
      let dw2_i =
        (impl_vol[i] - impl_vol[i - 1] - kappa_hat * (theta_hat - impl_vol[i - 1]) * delta)
          / (sigma_hat * impl_vol[i - 1].sqrt());

      sum_dw1dw2 += dw1_i * dw2_i;
    }

    let rho_hat = sum_dw1dw2 / (n as f64 * delta);

    self.initial_guess = Some(DVector::from_vec(vec![
      self.pricer.v0,
      theta_hat,
      rho_hat,
      kappa_hat,
      sigma_hat,
    ]));
  }
}

#[cfg(test)]
mod tests {
  use rand_distr::{Distribution, Normal};
  use stochastic_rs::{volatility::heston::Heston, Sampling2D};

  use super::*;

  #[test]
  fn test_heston_single_price() {
    let mut heston = HestonPricer {
      s0: 100.0,
      v0: 0.05,
      k: 100.0,
      r: 0.03,
      q: 0.02,
      rho: -0.8,
      kappa: 5.0,
      theta: 0.05,
      sigma: 0.5,
      lambda: Some(0.0),
      tau: Some(Either::Left(0.5)),
      ..Default::default()
    };

    let price = heston.calculate_price();

    match price {
      Either::Left((call, put)) => {
        println!("Call Price: {}, Put Price: {}", call, put);
      }
      _ => panic!("Expected a single price"),
    }
  }

  #[test]
  fn test_heston_multi_price() {
    let mut heston = HestonPricer {
      s0: 100.0,
      v0: 0.05,
      k: 100.0,
      r: 0.03,
      q: 0.02,
      rho: -0.8,
      kappa: 5.0,
      theta: 0.05,
      sigma: 0.5,
      lambda: Some(0.0),
      tau: Some(Either::Right(vec![0.5, 1.0, 2.0, 3.0])),
      ..Default::default()
    };

    let price = heston.calculate_price();

    match price {
      Either::Right(v) => {
        for (i, &(call, put)) in v.iter().enumerate() {
          println!(
            "Time to maturity {}: Call Price: {}, Put Price: {}",
            i + 1,
            call,
            put
          );
        }
      }
      _ => panic!("Expected multiple prices"),
    }
  }

  #[test]
  fn test_heston_calibrate() {
    // Calculate the initial guess for the calibration
    let pricer = HestonPricer::new(&HestonPricer {
      s0: 100.0,
      v0: 0.2,
      k: 100.0,
      r: 0.05,
      q: 0.02,
      lambda: Some(0.0),
      tau: Some(Either::Right(
        (0..=100)
          .map(|x| 0.5 + 0.1 * x as f64)
          .collect::<Vec<f64>>(),
      )),
      ..Default::default()
    });
    let yahoo = Yahoo::default();
    let model = Heston::new(&Heston {
      s0: Some(100.0),
      v0: Some(0.2),
      rho: -0.8,
      kappa: 1.0,
      theta: 0.25,
      sigma: 0.5,
      mu: 2.0,
      n: 1000,
      t: Some(1.0),
      use_sym: Some(true),
      ..Default::default()
    });
    let [s, v] = model.sample();
    let mut calibrator = HestonCalibrator::new(
      pricer,
      Some(yahoo),
      Some(v.to_vec()),
      Some(s.to_vec()),
      Some(
        Normal::new(25.0, 12.0)
          .unwrap()
          .sample_iter(rand::thread_rng())
          .take(1000)
          .collect::<Vec<f64>>(),
      ),
    );
    calibrator.calibrate();
  }
}