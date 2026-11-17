//! # Heston
//!
//! $$
//! \begin{aligned}dS_t&=\mu S_tdt+\sqrt{v_t}S_tdW_t^S\\dv_t&=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dW_t^v,\ d\langle W^S,W^v\rangle_t=\rho dt\end{aligned}
//! $$
//!
use std::f64::consts::FRAC_1_PI;
use std::ops::Range;

use implied_vol::DefaultSpecialFn;
use implied_vol::ImpliedBlackVolatility;
use num_complex::Complex64;
use stochastic_rs_distributions::special::norm_cdf;

use super::cf_quadrature::integrate_to_convergence;
use crate::OptionType;
use crate::pricing::digital::vanilla_payoff;
use crate::traits::ModelPricer;
use crate::traits::VanillaEuropeanCall;

mod greeks;
mod params;

pub use params::HestonParams;

/// Where the `P_j` integrals start: Gil–Pelaez's lower limit.
const LOWER_LIMIT: f64 = 1e-8;
/// The control variate leaves an integrand far below `P_j`'s own, whose panels pass a `1e-8` target
/// early: two bumped prices then differed by 7.5e-9 on a spot of 720, and by 3e-12 at `1e-10`.
const REST_TOLERANCE: f64 = 1e-10;
/// Past this variance the Gaussian, `1/√w` wide, is narrower than the quadrature's smallest
/// sub-panel: the control variate keeps this variance and no longer matches the mean.
const CONTROL_VARIANCE_CAP: f64 = 2e3;
/// `√w` at which the walk, which ends near `φ = 8,000`, stops inside the Gaussian, `1/√w` wide,
/// with an integrand it can no longer neglect.
const UNCOVERED: Range<f64> = 1e-5..1e-3;
/// Standard deviations `√w` from the money past which `P_j` is its Gaussian share to rounding.
const FAR: f64 = 60.0;

/// Heston (1993) semi-closed-form pricer; the query `(s, k, r, q, τ)` travels as arguments, so one
/// instance prices a whole strike/maturity grid.
/// Heston, S. L. (1993), Rev. Financ. Stud. 6(2), 327–343, DOI 10.1093/rfs/6.2.327.
///
/// ```
/// use stochastic_rs_quant::pricing::heston::HestonParams;
/// use stochastic_rs_quant::pricing::heston::HestonPricer;
/// use stochastic_rs_quant::traits::ModelPricer;
///
/// let params = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.7 };
/// let model = HestonPricer::new(params);
/// let atm = model.price_call(100.0, 100.0, 0.05, 0.02, 0.75);
/// let otm = model.price_call(100.0, 120.0, 0.05, 0.02, 0.75);
/// assert!(atm > otm);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct HestonPricer {
  params: HestonParams,
  /// Heston's market price of volatility risk `λ`.
  lambda: f64,
}

impl HestonPricer {
  /// Panics unless `params` passes [`HestonParams::new`].
  pub fn new(params: HestonParams) -> Self {
    Self {
      params: params.checked(),
      lambda: 0.0,
    }
  }

  /// Sets Heston's `λ` (default `0`); panics unless `lambda` is finite.
  pub fn with_vol_risk_premium(mut self, lambda: f64) -> Self {
    assert!(
      lambda.is_finite(),
      "lambda must satisfy `lambda.is_finite()`, got lambda = {lambda:?}"
    );
    self.lambda = lambda;
    self
  }

  pub fn params(&self) -> HestonParams {
    self.params
  }

  /// Call and put price at one query point: NaN for a negative or NaN `tau` or an `E[∫₀^τ v]` past
  /// `f64`; at `tau = 0` the payoff, which the `P_j` integrals, undamped there, do not resolve.
  ///
  /// $$
  /// C=Se^{-q\tau}P_1-Ke^{-r\tau}P_2,\qquad P=C+Ke^{-r\tau}-Se^{-q\tau}
  /// $$
  pub fn call_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> (f64, f64) {
    if tau.is_nan() || tau < 0.0 {
      return (f64::NAN, f64::NAN);
    }
    if tau == 0.0 {
      let payoff =
        |option_type| vanilla_payoff(option_type, s * (-q * tau).exp(), k * (-r * tau).exp());
      return (payoff(OptionType::Call), payoff(OptionType::Put));
    }
    let scale = self.rest_scale(tau, s, k, r, q);
    let call = s * (-q * tau).exp() * self.p(1, tau, (s, k, r, q), scale)
      - k * (-r * tau).exp() * self.p(2, tau, (s, k, r, q), scale);
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
  /// `∂C/∂v₀ = ∂P/∂v₀` under the Fourier integral as `D_j f_j`, no finite difference; NaN for
  /// `tau ≤ 0` or a NaN `tau`, as an expired option has a value and no sensitivity.
  pub fn call_put_initial_variance_vega(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
  ) -> (f64, f64) {
    if tau.is_nan() || tau <= 0.0 {
      return (f64::NAN, f64::NAN);
    }
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
      1 => self.params.kappa + self.lambda - self.params.rho * self.params.sigma,
      2 => self.params.kappa + self.lambda,
      _ => unreachable!("Heston P_j index must be 1 or 2"),
    }
  }

  /// `(f_j, D_j)` at `φ` in the Little-Trap form `g̃ = 1/g`, `e^{−dτ}`, which keeps the logarithm
  /// on its principal branch at any `τ`; `d` and `g̃` are evaluated once per sample.
  /// Albrecher, Mayer, Schoutens & Tistaert (2007), "The Little Heston Trap", Wilmott, 83–92.
  fn f_and_d(&self, j: u8, phi: f64, tau: f64, s: f64, r: f64, q: f64) -> (Complex64, Complex64) {
    let HestonParams {
      v0,
      kappa,
      theta,
      sigma,
      rho,
    } = self.params;
    let i = Complex64::i();
    let b = self.b(j);
    let d = ((b - rho * sigma * phi * i).powi(2)
      - sigma.powi(2) * (2.0 * i * self.u(j) * phi - phi.powi(2)))
    .sqrt();
    let drift = b - rho * sigma * i * phi - d;
    let g = drift / (b - rho * sigma * i * phi + d);
    let decay = (-d * tau).exp();
    let big_c = (r - q) * i * phi * tau
      + (kappa * theta / sigma.powi(2))
        * (drift * tau - 2.0 * ((1.0 - g * decay) / (1.0 - g)).ln());
    let big_d = (drift / sigma.powi(2)) * ((1.0 - decay) / (1.0 - g * decay));
    ((big_c + big_d * v0 + i * phi * s.ln()).exp(), big_d)
  }

  /// `E_j[∫₀^τ v]` under `P_j`'s mean reversion `b_j`: its Gaussian control variate's variance.
  fn mean_integrated_variance(&self, j: u8, tau: f64) -> f64 {
    let x = self.b(j) * tau;
    let (level, slope) = if x.abs() < 1e-4 {
      (1.0 - 0.5 * x, 0.5 - x / 6.0)
    } else {
      let level = -(-x).exp_m1() / x;
      (level, (1.0 - level) / x)
    };
    tau * (self.params.v0 * level + self.params.kappa * self.params.theta * tau * slope)
  }

  pub(self) fn re(
    &self,
    j: u8,
    tau: f64,
    (s, k, r, q): (f64, f64, f64, f64),
    control_variance: f64,
  ) -> impl Fn(f64) -> f64 {
    let self_ = *self;
    let ln_forward = s.ln() + (r - q) * tau;
    move |phi: f64| -> f64 {
      let i_phi = Complex64::i() * phi;
      let gaussian =
        (i_phi * ln_forward + (self_.u(j) * i_phi - 0.5 * phi * phi) * control_variance).exp();
      let (f, _) = self_.f_and_d(j, phi, tau, s, r, q);
      ((f - gaussian) * (-i_phi * k.ln()).exp() / i_phi).re
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
      let (f, big_d) = self_.f_and_d(j, phi, tau, s, r, q);
      (big_d * f * (-Complex64::i() * phi * k.ln()).exp() / (Complex64::i() * phi)).re
    }
  }

  /// The factor `φ` is rescaled by so that the walk ends past the Gaussian, one for both `P_j`;
  /// `None` where nothing is left to integrate, `FAR` from the money at a `√w` in `UNCOVERED`.
  fn rest_scale(&self, tau: f64, s: f64, k: f64, r: f64, q: f64) -> Option<f64> {
    let root = self.mean_integrated_variance(2, tau).sqrt();
    // A chf that outlasts the Gaussian (`v0 ≪ σ²τ`) has no such scale, and `φ` stays as it is.
    if !UNCOVERED.contains(&root) || self.f_and_d(2, 8.0 / root, tau, s, r, q).0.norm() >= 1e-8 {
      return Some(1.0);
    }
    let log_moneyness = (s / k).ln() + (r - q) * tau;
    (log_moneyness.abs() <= FAR * root).then_some(1e-2 / root)
  }

  /// Heston (1993)'s `P_j` less a Gaussian of variance `w = E_j[∫v]`, whose share is `N(d_j)`, a
  /// step at `w = 0`: the rest vanishes at `φ → 0`, and almost everywhere at a `τ` of seconds.
  pub(self) fn p(
    &self,
    j: u8,
    tau: f64,
    (s, k, r, q): (f64, f64, f64, f64),
    scale: Option<f64>,
  ) -> f64 {
    let expected = self.mean_integrated_variance(j, tau);
    if !expected.is_finite() {
      return f64::NAN;
    }
    let w = expected.min(CONTROL_VARIANCE_CAP);
    let log_moneyness = (s / k).ln() + (r - q) * tau + self.u(j) * w;
    let d = if log_moneyness == 0.0 {
      0.0
    } else {
      log_moneyness / w.sqrt()
    };
    let Some(scale) = scale else {
      return norm_cdf(d);
    };
    let rest = self.re(j, tau, (s, k, r, q), w);
    let scaled = |y: f64| scale * rest(scale * y);
    norm_cdf(d) + FRAC_1_PI * integrate_to_convergence(scaled, LOWER_LIMIT / scale, REST_TOLERANCE)
  }

  fn p_initial_variance_derivative(&self, j: u8, tau: f64, s: f64, k: f64, r: f64, q: f64) -> f64 {
    FRAC_1_PI
      * integrate_to_convergence(
        self.re_initial_variance_derivative(j, tau, s, k, r, q),
        LOWER_LIMIT,
        1e-8,
      )
  }
}

#[cfg(test)]
mod tests;
