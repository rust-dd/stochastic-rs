//! # Bjerksund-Stensland 2002 American Option Approximation
//!
//! Analytical approximation for American call and put options using a
//! flat early-exercise boundary split at $t_1=\tfrac12(\sqrt5-1)T$:
//!
//! $$
//! C_{\mathrm{am}}=\alpha_2 F^{\beta}
//!   -\alpha_2\,\phi(F,t_1,\beta,I_2,I_2)
//!   +\phi(F,t_1,1,I_2,I_2)
//!   -\phi(F,t_1,1,I_1,I_2)
//!   -X\,\phi(F,t_1,0,I_2,I_2)
//!   +X\,\phi(F,t_1,0,I_1,I_2)
//!   +\cdots
//! $$
//!
//! Put values use the Bjerksund-Stensland symmetry relation.
//!
//! Reference: Bjerksund, P. & Stensland, G. (2002). "Closed Form Valuation
//! of American Options." Discussion paper 2002/09, NHH.
//! <https://www.researchgate.net/publication/228801918>

use owens_t::biv_norm;
use stochastic_rs_distributions::special::norm_cdf;

use crate::OptionType;
use crate::traits::PricerExt;
use crate::traits::TimeExt;

/// Bjerksund-Stensland 2002 pricer for American options.
///
/// Falls back to the GBS (European) value when early exercise
/// is never optimal (i.e. when `b >= r` for calls).
#[derive(Debug, Clone)]
pub struct BjerksundStensland2002Pricer {
  /// Spot / forward price
  pub s: f64,
  /// Volatility
  pub v: f64,
  /// Strike price
  pub k: f64,
  /// Risk-free rate
  pub r: f64,
  /// Dividend yield
  pub q: Option<f64>,
  /// Time to maturity in years
  pub tau: Option<f64>,
  /// Evaluation date
  pub eval: Option<chrono::NaiveDate>,
  /// Expiration date
  pub expiration: Option<chrono::NaiveDate>,
  /// Option type
  pub option_type: OptionType,
}

impl BjerksundStensland2002Pricer {
  pub fn new(
    s: f64,
    v: f64,
    k: f64,
    r: f64,
    q: Option<f64>,
    tau: Option<f64>,
    eval: Option<chrono::NaiveDate>,
    expiration: Option<chrono::NaiveDate>,
    option_type: OptionType,
  ) -> Self {
    Self {
      s,
      v,
      k,
      r,
      q,
      tau,
      eval,
      expiration,
      option_type,
    }
  }

  pub fn builder(s: f64, v: f64, k: f64, r: f64) -> BjerksundStensland2002PricerBuilder {
    BjerksundStensland2002PricerBuilder {
      s,
      v,
      k,
      r,
      q: None,
      tau: None,
      eval: None,
      expiration: None,
      option_type: OptionType::Call,
    }
  }

  /// Cost of carry: b = r - q
  fn b(&self) -> f64 {
    self.r - self.q.unwrap_or(0.0)
  }

  /// European GBS call price (used as lower bound).
  fn gbs_call(&self, fs: f64, x: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = ((fs / x).ln() + (b + 0.5 * v * v) * t) / (v * t.sqrt());
    let d2 = d1 - v * t.sqrt();
    fs * ((b - r) * t).exp() * norm_cdf(d1) - x * (-r * t).exp() * norm_cdf(d2)
  }

  /// European GBS put price.
  fn gbs_put(&self, fs: f64, x: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = ((fs / x).ln() + (b + 0.5 * v * v) * t) / (v * t.sqrt());
    let d2 = d1 - v * t.sqrt();
    x * (-r * t).exp() * norm_cdf(-d2) - fs * ((b - r) * t).exp() * norm_cdf(-d1)
  }

  /// The $\phi$ intermediate function.
  fn phi(&self, fs: f64, t: f64, gamma: f64, h: f64, i: f64, r: f64, b: f64, v: f64) -> f64 {
    let v2 = v * v;
    let d1 = -((fs / h).ln() + (b + (gamma - 0.5) * v2) * t) / (v * t.sqrt());
    let d2 = d1 - 2.0 * (i / fs).ln() / (v * t.sqrt());

    let lambda = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * v2;
    let kappa = 2.0 * b / v2 + (2.0 * gamma - 1.0);

    (lambda * t).exp() * fs.powf(gamma) * (norm_cdf(d1) - (i / fs).powf(kappa) * norm_cdf(d2))
  }

  /// The $\psi$ intermediate function (uses bivariate normal CDF).
  fn psi(
    &self,
    fs: f64,
    t2: f64,
    gamma: f64,
    h: f64,
    i2: f64,
    i1: f64,
    t1: f64,
    r: f64,
    b: f64,
    v: f64,
  ) -> f64 {
    let v2 = v * v;
    let vsqrt_t1 = v * t1.sqrt();
    let vsqrt_t2 = v * t2.sqrt();

    let bgamma_t1 = (b + (gamma - 0.5) * v2) * t1;
    let bgamma_t2 = (b + (gamma - 0.5) * v2) * t2;

    let d1 = ((fs / i1).ln() + bgamma_t1) / vsqrt_t1;
    let d2 = ((i2 * i2 / (fs * i1)).ln() + bgamma_t1) / vsqrt_t1;
    let d3 = ((fs / i1).ln() - bgamma_t1) / vsqrt_t1;
    let d4 = ((i2 * i2 / (fs * i1)).ln() - bgamma_t1) / vsqrt_t1;

    let e1 = ((fs / h).ln() + bgamma_t2) / vsqrt_t2;
    let e2 = ((i2 * i2 / (fs * h)).ln() + bgamma_t2) / vsqrt_t2;
    let e3 = ((i1 * i1 / (fs * h)).ln() + bgamma_t2) / vsqrt_t2;
    let e4 = ((fs * i1 * i1 / (h * i2 * i2)).ln() + bgamma_t2) / vsqrt_t2;

    let tau = (t1 / t2).sqrt();
    let lambda = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * v2;
    let kappa = 2.0 * b / v2 + (2.0 * gamma - 1.0);

    // owens_t::biv_norm computes P(X > x, Y > y), so negate args for CDF
    let cbnd = |a: f64, b: f64, rho: f64| -> f64 { biv_norm(-a, -b, rho) };

    (lambda * t2).exp()
      * fs.powf(gamma)
      * (cbnd(-d1, -e1, tau)
        - (i2 / fs).powf(kappa) * cbnd(-d2, -e2, tau)
        - (i1 / fs).powf(kappa) * cbnd(-d3, -e3, -tau)
        + (i1 / i2).powf(kappa) * cbnd(-d4, -e4, -tau))
  }

  /// Core BS2002 call pricing (works on transformed inputs for puts).
  fn bs2002_call(&self, fs: f64, x: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
    let e_value = self.gbs_call(fs, x, t, r, b, v);

    // If b >= r, early exercise is never optimal
    if b >= r {
      return e_value;
    }

    let v2 = v * v;
    let t1 = 0.5 * (5.0_f64.sqrt() - 1.0) * t;
    let t2 = t;

    let beta_inside = ((b / v2 - 0.5).powi(2) + 2.0 * r / v2).abs();
    let beta = (0.5 - b / v2) + beta_inside.sqrt();
    let b_infinity = (beta / (beta - 1.0)) * x;
    let b_zero = f64::max(x, (r / (r - b)) * x);

    let h1 = -(b * t1 + 2.0 * v * t1.sqrt()) * (x * x / ((b_infinity - b_zero) * b_zero));
    let h2 = -(b * t2 + 2.0 * v * t2.sqrt()) * (x * x / ((b_infinity - b_zero) * b_zero));

    let i1 = b_zero + (b_infinity - b_zero) * (1.0 - h1.exp());
    let i2 = b_zero + (b_infinity - b_zero) * (1.0 - h2.exp());

    let alpha1 = (i1 - x) * i1.powf(-beta);
    let alpha2 = (i2 - x) * i2.powf(-beta);

    // Check for immediate exercise
    if fs >= i2 {
      return fs - x;
    }

    let value = alpha2 * fs.powf(beta) - alpha2 * self.phi(fs, t1, beta, i2, i2, r, b, v)
      + self.phi(fs, t1, 1.0, i2, i2, r, b, v)
      - self.phi(fs, t1, 1.0, i1, i2, r, b, v)
      - x * self.phi(fs, t1, 0.0, i2, i2, r, b, v)
      + x * self.phi(fs, t1, 0.0, i1, i2, r, b, v)
      + alpha1 * self.phi(fs, t1, beta, i1, i2, r, b, v)
      - alpha1 * self.psi(fs, t2, beta, i1, i2, i1, t1, r, b, v)
      + self.psi(fs, t2, 1.0, i1, i2, i1, t1, r, b, v)
      - self.psi(fs, t2, 1.0, x, i2, i1, t1, r, b, v)
      - x * self.psi(fs, t2, 0.0, i1, i2, i1, t1, r, b, v)
      + x * self.psi(fs, t2, 0.0, x, i2, i1, t1, r, b, v);

    // Ensure at least the European value
    f64::max(value, e_value)
  }
}

#[derive(Debug, Clone)]
pub struct BjerksundStensland2002PricerBuilder {
  s: f64,
  v: f64,
  k: f64,
  r: f64,
  q: Option<f64>,
  tau: Option<f64>,
  eval: Option<chrono::NaiveDate>,
  expiration: Option<chrono::NaiveDate>,
  option_type: OptionType,
}

impl BjerksundStensland2002PricerBuilder {
  pub fn q(mut self, q: f64) -> Self {
    self.q = Some(q);
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
  pub fn expiration(mut self, expiration: chrono::NaiveDate) -> Self {
    self.expiration = Some(expiration);
    self
  }
  pub fn option_type(mut self, option_type: OptionType) -> Self {
    self.option_type = option_type;
    self
  }
  pub fn build(self) -> BjerksundStensland2002Pricer {
    BjerksundStensland2002Pricer {
      s: self.s,
      v: self.v,
      k: self.k,
      r: self.r,
      q: self.q,
      tau: self.tau,
      eval: self.eval,
      expiration: self.expiration,
      option_type: self.option_type,
    }
  }
}

impl PricerExt for BjerksundStensland2002Pricer {
  fn calculate_call_put(&self) -> (f64, f64) {
    let tau = self.tau_or_from_dates();
    let b = self.b();

    // Call: direct BS2002
    let call = self.bs2002_call(self.s, self.k, tau, self.r, b, self.v);

    // Put: use the Bjerksund-Stensland symmetry relation
    // P(S, X, T, r, b, v) = C(X, S, T, r-b, -b, v)
    let put_as_call = self.bs2002_call(self.k, self.s, tau, self.r - b, -b, self.v);
    let put = f64::max(
      put_as_call,
      self.gbs_put(self.s, self.k, tau, self.r, b, self.v),
    );

    (call, put)
  }

  fn calculate_price(&self) -> f64 {
    let (call, put) = self.calculate_call_put();
    match self.option_type {
      OptionType::Call => call,
      OptionType::Put => put,
    }
  }
}

impl TimeExt for BjerksundStensland2002Pricer {
  fn tau(&self) -> Option<f64> {
    self.tau
  }

  fn eval(&self) -> Option<chrono::NaiveDate> {
    self.eval
  }

  fn expiration(&self) -> Option<chrono::NaiveDate> {
    self.expiration
  }
}
