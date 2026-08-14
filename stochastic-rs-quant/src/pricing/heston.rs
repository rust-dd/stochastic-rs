//! # Heston
//!
//! $$
//! \begin{aligned}dS_t&=\mu S_tdt+\sqrt{v_t}S_tdW_t^S\\dv_t&=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dW_t^v,\ d\langle W^S,W^v\rangle_t=\rho dt\end{aligned}
//! $$
//!
use std::f64::consts::FRAC_1_PI;

use implied_vol::DefaultSpecialFn;
use implied_vol::ImpliedBlackVolatility;
use num_complex::Complex64;

use super::cf_quadrature::integrate_to_convergence;
use crate::OptionType;
use crate::traits::GreeksExt;
use crate::traits::PricerExt;
use crate::traits::TimeExt;

/// Heston stochastic volatility pricer using the characteristic-function method.
///
/// Source:
/// - Heston, S. L. (1993), "A Closed-Form Solution for Options with Stochastic Volatility
///   with Applications to Bond and Currency Options"
///   https://doi.org/10.1093/rfs/6.2.327
#[derive(Clone)]
pub struct HestonPricer {
  /// Stock price
  pub s: f64,
  /// Initial variance v₀ — a variance, not a volatility: this file's own
  /// `GreeksExt` impl documents the chain rule `σ = √v0` it uses to convert
  /// into volatility-space derivatives (see `vega`/`vanna`/`volga`/`veta`).
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
  /// Long-run variance level (θ) — a variance, not a volatility, for the
  /// same reason as `v0`.
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
  pub expiration: Option<chrono::NaiveDate>,
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
    expiration: Option<chrono::NaiveDate>,
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
      expiration,
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
      expiration: None,
    }
  }
}

#[derive(Debug, Clone)]
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
  expiration: Option<chrono::NaiveDate>,
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
  pub fn expiration(mut self, expiration: chrono::NaiveDate) -> Self {
    self.expiration = Some(expiration);
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
      expiration: self.expiration,
    }
  }
}

impl PricerExt for HestonPricer {
  fn calculate_call_put(&self) -> (f64, f64) {
    let tau = self.tau_or_from_dates();

    let call = self.s * (-self.q.unwrap_or(0.0) * tau).exp() * self.p(1, tau)
      - self.k * (-self.r * tau).exp() * self.p(2, tau);
    let put = call + self.k * (-self.r * tau).exp() - self.s * (-self.q.unwrap_or(0.0) * tau).exp();

    (call, put)
  }

  fn calculate_price(&self) -> f64 {
    self.calculate_call_put().0
  }

  fn implied_volatility(&self, c_price: f64, option_type: OptionType) -> f64 {
    let tau = self.calculate_tau_in_years();
    let q = self.q.unwrap_or(0.0);
    let forward = self.s * ((self.r - q) * tau).exp();
    let undiscounted_price = c_price * (self.r * tau).exp();
    ImpliedBlackVolatility::builder()
      .option_price(undiscounted_price)
      .forward(forward)
      .strike(self.k)
      .expiry(tau)
      .is_call(option_type == OptionType::Call)
      .build()
      .and_then(|iv| iv.calculate::<DefaultSpecialFn>())
      .unwrap_or(f64::NAN)
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
    self.expiration
  }
}

impl HestonPricer {
  /// Returns analytic call and put derivatives with respect to initial variance.
  ///
  /// For each Heston probability, the derivative is evaluated under the
  /// Fourier integral as `D_j f_j`. This avoids subtracting nearly equal option
  /// prices in a finite difference. Put-call parity makes the two derivatives
  /// identical when rates and dividend yield do not depend on initial variance.
  pub fn calculate_call_put_initial_variance_vega(&self) -> (f64, f64) {
    let tau = self.tau_or_from_dates();
    let vega =
      self.s * (-self.q.unwrap_or(0.0) * tau).exp() * self.p_initial_variance_derivative(1, tau)
        - self.k * (-self.r * tau).exp() * self.p_initial_variance_derivative(2, tau);
    (vega, vega)
  }

  pub(self) fn u(&self, j: u8) -> f64 {
    match j {
      1 => 0.5,
      2 => -0.5,
      _ => unreachable!("Heston P_j index must be 1 or 2"),
    }
  }

  pub(self) fn b(&self, j: u8) -> f64 {
    match j {
      1 => self.kappa + self.lambda.unwrap_or(0.0) - self.rho * self.sigma,
      2 => self.kappa + self.lambda.unwrap_or(0.0),
      _ => unreachable!("Heston P_j index must be 1 or 2"),
    }
  }

  pub(self) fn d(&self, j: u8, phi: f64) -> Complex64 {
    ((self.b(j) - self.rho * self.sigma * phi * Complex64::i()).powi(2)
      - self.sigma.powi(2) * (2.0 * Complex64::i() * self.u(j) * phi - phi.powi(2)))
    .sqrt()
  }

  /// Albrecher-Mayer-Schoutens-Tistaert (2007) "Little Heston Trap" form:
  /// g̃ = 1/g_original keeps log-argument on the principal branch for all τ.
  pub(self) fn g(&self, j: u8, phi: f64) -> Complex64 {
    (self.b(j) - self.rho * self.sigma * Complex64::i() * phi - self.d(j, phi))
      / (self.b(j) - self.rho * self.sigma * Complex64::i() * phi + self.d(j, phi))
  }

  pub(self) fn C(&self, j: u8, phi: f64, tau: f64) -> Complex64 {
    (self.r - self.q.unwrap_or(0.0)) * Complex64::i() * phi * tau
      + (self.kappa * self.theta / self.sigma.powi(2))
        * ((self.b(j) - self.rho * self.sigma * Complex64::i() * phi - self.d(j, phi)) * tau
          - 2.0
            * ((1.0 - self.g(j, phi) * (-self.d(j, phi) * tau).exp()) / (1.0 - self.g(j, phi)))
              .ln())
  }

  pub(self) fn D(&self, j: u8, phi: f64, tau: f64) -> Complex64 {
    ((self.b(j) - self.rho * self.sigma * Complex64::i() * phi - self.d(j, phi))
      / self.sigma.powi(2))
      * ((1.0 - (-self.d(j, phi) * tau).exp())
        / (1.0 - self.g(j, phi) * (-self.d(j, phi) * tau).exp()))
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

  fn re_initial_variance_derivative(&self, j: u8, tau: f64) -> impl Fn(f64) -> f64 {
    let self_ = self.clone();
    move |phi: f64| -> f64 {
      (self_.D(j, phi, tau) * self_.f(j, phi, tau) * (-Complex64::i() * phi * self_.k.ln()).exp()
        / (Complex64::i() * phi))
        .re
    }
  }

  /// Risk-neutral probability integral `P_j` in the original Heston semi-closed form.
  ///
  /// Source:
  /// - Heston, S. L. (1993)
  ///   https://doi.org/10.1093/rfs/6.2.327
  pub(self) fn p(&self, j: u8, tau: f64) -> f64 {
    0.5 + FRAC_1_PI * integrate_to_convergence(self.re(j, tau), 0.00001, 1e-8)
  }

  fn p_initial_variance_derivative(&self, j: u8, tau: f64) -> f64 {
    FRAC_1_PI * integrate_to_convergence(self.re_initial_variance_derivative(j, tau), 0.00001, 1e-8)
  }
}

impl HestonPricer {
  fn h_s(&self) -> f64 {
    self.s.abs() * 1e-4
  }

  fn h_v(&self) -> f64 {
    self.v0.abs().max(0.01) * 1e-4
  }

  const H_TAU: f64 = 1e-5;
  const H_R: f64 = 1e-5;

  /// Clone with `s`/`v0`/`tau`/`r` bumped. `v0` is floored at `1e-12`
  /// (mirrors [`AnalyticHestonEngine`](crate::pricing::engines::AnalyticHestonEngine)'s
  /// own down-bump clamp) so a downward variance bump near zero cannot
  /// produce a negative, model-invalid variance.
  fn bumped(&self, ds: f64, dv0: f64, dtau: f64, dr: f64) -> Self {
    let mut p = self.clone();
    p.s += ds;
    p.v0 = (p.v0 + dv0).max(1e-12);
    let tau = p.tau_or_from_dates();
    p.tau = Some(tau + dtau);
    p.eval = None;
    p.expiration = None;
    p.r += dr;
    p
  }

  /// `∂(call price)/∂v0`, analytic (no finite difference) via
  /// [`calculate_call_put_initial_variance_vega`](Self::calculate_call_put_initial_variance_vega).
  /// Identical for call and put per that method's own doc.
  fn v0_vega(&self) -> f64 {
    self.calculate_call_put_initial_variance_vega().0
  }
}

/// Central finite-difference Greeks for the Heston (1993) semi-closed-form
/// call price (the same price [`PricerExt::calculate_price`] returns —
/// this pricer has no separate put path to differentiate).
///
/// `vega`/`vanna`/`volga`/`veta` bump the variance parameter `v0` — not
/// `√v0` — mirroring
/// [`AnalyticHestonEngine::finite_diff_greeks`](crate::pricing::engines::AnalyticHestonEngine),
/// then convert to a volatility-space derivative via the chain rule
/// `σ = √v0`:
/// `∂P/∂σ = 2√v0 · ∂P/∂v0` and its higher partials. The `∂P/∂v0` building
/// block itself is the analytic
/// [`calculate_call_put_initial_variance_vega`](HestonPricer::calculate_call_put_initial_variance_vega)
/// rather than a finite difference, for precision — vanna/volga/veta then
/// finite-difference *that* analytic function instead of double
/// finite-differencing the raw price.
///
/// `theta`/`charm`/`veta` use the calendar `-∂/∂τ` convention mandated by
/// [`GreeksExt::theta`]'s own doc (`∂V/∂t`) and matching
/// [`BSMPricer`](crate::pricing::bsm::BSMPricer)'s / `Merton1976Pricer`'s
/// Greeks — the negative of the raw `+∂P/∂τ` that
/// [`AnalyticHestonEngine::finite_diff_greeks`](crate::pricing::engines::AnalyticHestonEngine)
/// computes (that engine predates this trait impl and was never updated to
/// match; see `heston/tests.rs::heston_greeks_match_engine_bumps` for how
/// the two are reconciled in tests).
///
/// **`NaN` is a deliberate return value here, not just the trait default's
/// "unimplemented" marker.** `vega`/`vanna`/`volga`/`veta` divide through
/// the `σ = √v0` chain rule above, which is undefined at `v0 <= 0`; each
/// guards that case explicitly and returns `NaN` rather than a wrong
/// finite number. `theta`/`charm`/`veta` carry a second, independent guard
/// on `tau_or_from_dates()`, returning `NaN` when it is non-finite or not
/// safely larger than the central-difference step `H_TAU` — see
/// `heston_greeks_nan_at_degenerate_inputs` for both guards exercised
/// directly.
impl GreeksExt for HestonPricer {
  fn delta(&self) -> f64 {
    let h = self.h_s();
    (self.bumped(h, 0.0, 0.0, 0.0).calculate_price()
      - self.bumped(-h, 0.0, 0.0, 0.0).calculate_price())
      / (2.0 * h)
  }

  fn gamma(&self) -> f64 {
    let h = self.h_s();
    let p0 = self.calculate_price();
    (self.bumped(h, 0.0, 0.0, 0.0).calculate_price() - 2.0 * p0
      + self.bumped(-h, 0.0, 0.0, 0.0).calculate_price())
      / (h * h)
  }

  fn vega(&self) -> f64 {
    if self.v0 <= 0.0 {
      return f64::NAN;
    }
    2.0 * self.v0.sqrt() * self.v0_vega()
  }

  fn theta(&self) -> f64 {
    let tau = self.tau_or_from_dates();
    let h = Self::H_TAU;
    if !(tau.is_finite() && tau > h) {
      return f64::NAN;
    }
    -(self.bumped(0.0, 0.0, h, 0.0).calculate_price()
      - self.bumped(0.0, 0.0, -h, 0.0).calculate_price())
      / (2.0 * h)
  }

  fn rho(&self) -> f64 {
    let h = Self::H_R;
    (self.bumped(0.0, 0.0, 0.0, h).calculate_price()
      - self.bumped(0.0, 0.0, 0.0, -h).calculate_price())
      / (2.0 * h)
  }

  fn vanna(&self) -> f64 {
    if self.v0 <= 0.0 {
      return f64::NAN;
    }
    let h = self.h_s();
    let p_s_v0 = (self.bumped(h, 0.0, 0.0, 0.0).v0_vega()
      - self.bumped(-h, 0.0, 0.0, 0.0).v0_vega())
      / (2.0 * h);
    2.0 * self.v0.sqrt() * p_s_v0
  }

  fn charm(&self) -> f64 {
    let tau = self.tau_or_from_dates();
    let ht = Self::H_TAU;
    if !(tau.is_finite() && tau > ht) {
      return f64::NAN;
    }
    let hs = self.h_s();
    -(self.bumped(hs, 0.0, ht, 0.0).calculate_price()
      - self.bumped(hs, 0.0, -ht, 0.0).calculate_price()
      - self.bumped(-hs, 0.0, ht, 0.0).calculate_price()
      + self.bumped(-hs, 0.0, -ht, 0.0).calculate_price())
      / (4.0 * hs * ht)
  }

  fn volga(&self) -> f64 {
    if self.v0 <= 0.0 {
      return f64::NAN;
    }
    let h = self.h_v();
    let p_v0v0 = (self.bumped(0.0, h, 0.0, 0.0).v0_vega()
      - self.bumped(0.0, -h, 0.0, 0.0).v0_vega())
      / (2.0 * h);
    4.0 * self.v0 * p_v0v0 + 2.0 * self.v0_vega()
  }

  fn veta(&self) -> f64 {
    if self.v0 <= 0.0 {
      return f64::NAN;
    }
    let tau = self.tau_or_from_dates();
    let h = Self::H_TAU;
    if !(tau.is_finite() && tau > h) {
      return f64::NAN;
    }
    let p_tau_v0 = (self.bumped(0.0, 0.0, h, 0.0).v0_vega()
      - self.bumped(0.0, 0.0, -h, 0.0).v0_vega())
      / (2.0 * h);
    -2.0 * self.v0.sqrt() * p_tau_v0
  }
}

#[cfg(test)]
mod tests;
