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
use crate::traits::ModelPricer;
use crate::traits::VanillaEuropeanCall;

mod greeks;

/// Heston stochastic volatility pricer using the characteristic-function method.
///
/// The struct holds **model state only** — the six Heston parameters. Spot,
/// strike, rate, dividend yield and maturity are the pricing *query* and
/// travel as arguments to [`ModelPricer::price_call`] and to every Greek,
/// so one instance prices a whole strike/maturity grid.
///
/// Source:
/// - Heston, S. L. (1993), "A Closed-Form Solution for Options with Stochastic Volatility
///   with Applications to Bond and Currency Options"
///   https://doi.org/10.1093/rfs/6.2.327
///
/// ```
/// use stochastic_rs_quant::pricing::heston::HestonPricer;
/// use stochastic_rs_quant::traits::ModelPricer;
///
/// let model = HestonPricer::new(0.04, -0.7, 2.0, 0.04, 0.3, None);
/// let atm = model.price_call(100.0, 100.0, 0.05, 0.02, 0.75);
/// let otm = model.price_call(100.0, 120.0, 0.05, 0.02, 0.75);
/// assert!(atm > otm);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct HestonPricer {
  /// Initial variance v₀ — a variance, not a volatility: this module's own
  /// Greeks document the chain rule `σ = √v0` they use to convert into
  /// volatility-space derivatives (see `vega`/`vanna`/`volga`/`veta`).
  pub v0: f64,
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
}

impl HestonPricer {
  pub const fn new(
    v0: f64,
    rho: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    lambda: Option<f64>,
  ) -> Self {
    Self {
      v0,
      rho,
      kappa,
      theta,
      sigma,
      lambda,
    }
  }

  /// Call and put price at one query point.
  ///
  /// $$
  /// C=Se^{-q\tau}P_1-Ke^{-r\tau}P_2,\qquad P=C+Ke^{-r\tau}-Se^{-q\tau}
  /// $$
  pub fn call_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> (f64, f64) {
    let call = s * (-q * tau).exp() * self.p(1, tau, s, k, r, q)
      - k * (-r * tau).exp() * self.p(2, tau, s, k, r, q);
    let put = call + k * (-r * tau).exp() - s * (-q * tau).exp();

    (call, put)
  }

  /// Black volatility implied by `price` at one query point.
  ///
  /// Depends on none of this model's own parameters — it inverts a price
  /// for a volatility rather than pricing at one — but is kept here as the
  /// inverse of [`call_put`](Self::call_put), sharing its `b = r - q`
  /// carry convention.
  ///
  /// Returns [`f64::NAN`] when the price is outside the no-arbitrage bounds
  /// the inversion can invert.
  pub fn implied_volatility(
    &self,
    c_price: f64,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    option_type: OptionType,
  ) -> f64 {
    let forward = s * ((r - q) * tau).exp();
    let undiscounted_price = c_price * (r * tau).exp();
    ImpliedBlackVolatility::builder()
      .option_price(undiscounted_price)
      .forward(forward)
      .strike(k)
      .expiry(tau)
      .is_call(option_type == OptionType::Call)
      .build()
      .and_then(|iv| iv.calculate::<DefaultSpecialFn>())
      .unwrap_or(f64::NAN)
  }
}

impl ModelPricer for HestonPricer {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put(s, k, r, q, tau).0
  }

  /// Takes the closed form rather than the trait's vanilla-parity default.
  /// The two are *mathematically* the same here — this model's carry factor
  /// really is $e^{-q\tau}$ — but the default associates the same three
  /// terms in a different order, so it can land an ulp away from the value
  /// the pre-query `calculate_call_put().1` returned. See
  /// `heston_price_put_matches_parity_but_is_the_closed_form`.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put(s, k, r, q, tau).1
  }
}

/// European vanilla call on the forward $Se^{(r-q)\tau}$ — this model's
/// cost of carry is $b=r-q$, so the default forward applies unchanged.
impl VanillaEuropeanCall for HestonPricer {}

impl HestonPricer {
  /// Returns analytic call and put derivatives with respect to initial variance.
  ///
  /// For each Heston probability, the derivative is evaluated under the
  /// Fourier integral as `D_j f_j`. This avoids subtracting nearly equal option
  /// prices in a finite difference. Put-call parity makes the two derivatives
  /// identical when rates and dividend yield do not depend on initial variance.
  pub fn call_put_initial_variance_vega(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
  ) -> (f64, f64) {
    let vega = s * (-q * tau).exp() * self.p_initial_variance_derivative(1, tau, s, k, r, q)
      - k * (-r * tau).exp() * self.p_initial_variance_derivative(2, tau, s, k, r, q);
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

  pub(self) fn C(&self, j: u8, phi: f64, tau: f64, r: f64, q: f64) -> Complex64 {
    (r - q) * Complex64::i() * phi * tau
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

  pub(self) fn f(&self, j: u8, phi: f64, tau: f64, s: f64, r: f64, q: f64) -> Complex64 {
    (self.C(j, phi, tau, r, q) + self.D(j, phi, tau) * self.v0 + Complex64::i() * phi * s.ln())
      .exp()
  }

  pub(self) fn re(&self, j: u8, tau: f64, s: f64, k: f64, r: f64, q: f64) -> impl Fn(f64) -> f64 {
    let self_ = *self;
    move |phi: f64| -> f64 {
      (self_.f(j, phi, tau, s, r, q) * (-Complex64::i() * phi * k.ln()).exp()
        / (Complex64::i() * phi))
        .re
    }
  }

  fn re_initial_variance_derivative(
    &self,
    j: u8,
    tau: f64,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
  ) -> impl Fn(f64) -> f64 {
    let self_ = *self;
    move |phi: f64| -> f64 {
      (self_.D(j, phi, tau)
        * self_.f(j, phi, tau, s, r, q)
        * (-Complex64::i() * phi * k.ln()).exp()
        / (Complex64::i() * phi))
        .re
    }
  }

  /// Risk-neutral probability integral `P_j` in the original Heston semi-closed form.
  ///
  /// Source:
  /// - Heston, S. L. (1993)
  ///   https://doi.org/10.1093/rfs/6.2.327
  pub(self) fn p(&self, j: u8, tau: f64, s: f64, k: f64, r: f64, q: f64) -> f64 {
    0.5 + FRAC_1_PI * integrate_to_convergence(self.re(j, tau, s, k, r, q), 0.00001, 1e-8)
  }

  fn p_initial_variance_derivative(&self, j: u8, tau: f64, s: f64, k: f64, r: f64, q: f64) -> f64 {
    FRAC_1_PI
      * integrate_to_convergence(
        self.re_initial_variance_derivative(j, tau, s, k, r, q),
        0.00001,
        1e-8,
      )
  }
}

#[cfg(test)]
mod tests;
