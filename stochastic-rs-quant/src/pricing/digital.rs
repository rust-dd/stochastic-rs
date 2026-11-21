//! Cash-or-nothing, asset-or-nothing, gap and supershare digitals under Black–Scholes.
//!
//! $$
//! C_{\text{CoN}}=Qe^{-rT}N(d_2),\quad C_{\text{AoN}}=Se^{(b-r)T}N(d_1),\quad
//! d_{1,2}=\frac{\ln(S/K)+(b\pm\tfrac12\sigma^2)T}{\sigma\sqrt T}
//! $$
//!
//! The carry `b = r − q` is each query's own (Merton 1973), never a field frozen at one `(r, q)`.
//! Reference: Reiner & Rubinstein (1991), "Unscrambling the Binary Code"
//! Reference: Haug (2007), The Complete Guide to Option Pricing Formulas, Ch. 4, ISBN 0071389970
//! Reference: Hull (2018), Options, Futures, and Other Derivatives, 10th ed., §26.9

use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_distributions::special::norm_pdf;

use crate::OptionType;
use crate::traits::Greeks;
use crate::traits::ModelPricer;
use crate::traits::RealExt;

/// Cash-or-nothing digital: pays the cash $Q$ when the option finishes strictly in the money, so
/// nothing at `s = k`; at `tau = 0` it is worth that payoff.
///
/// $$
/// C_{\text{CoN}}=Qe^{-rT}N(d_2),\qquad P_{\text{CoN}}=Qe^{-rT}N(-d_2)
/// $$
///
/// ```
/// use stochastic_rs_quant::pricing::CashOrNothingPricer;
/// use stochastic_rs_quant::traits::ModelPricer;
///
/// let model = CashOrNothingPricer::new(10.0, 0.35);
/// let itm = model.price_call(100.0, 80.0, 0.06, 0.0, 0.75);
/// let otm = model.price_call(100.0, 120.0, 0.06, 0.0, 0.75);
/// assert!(itm > otm);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CashOrNothingPricer {
  /// Cash payout `Q`, a term of the contract.
  pub cash: f64,
  /// Annualised Black–Scholes volatility.
  pub sigma: f64,
}

impl CashOrNothingPricer {
  /// Panics for a negative or infinite `sigma` (a negative one flips `d1` into a finite wrong
  /// price) and an infinite `cash`; a NaN is kept, as an unlinked engine handle must price NaN.
  pub fn new(cash: f64, sigma: f64) -> Self {
    assert_digital_sigma(sigma);
    assert!(
      !cash.is_infinite(),
      "cash must satisfy `!cash.is_infinite()`, got cash = {cash:?}"
    );
    Self { cash, sigma }
  }

  /// Every Greek this pricer exposes at one query point, in a [`Greeks`] aggregate; the six it
  /// does not expose stay [`f64::NAN`].
  pub fn greeks(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    option_type: OptionType,
  ) -> Greeks {
    if tau == 0.0 {
      return Greeks::nan();
    }
    Greeks {
      delta: self.delta(s, k, r, q, tau, option_type),
      gamma: self.gamma(s, k, r, q, tau, option_type),
      vega: self.vega(s, k, r, q, tau, option_type),
      ..Greeks::nan()
    }
  }

  /// Delta — $\partial V/\partial S$.
  pub fn delta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    let (_, d2) = bsm_d1_d2(s, k, r - q, self.sigma, tau);
    let disc = (-r * tau).exp();
    let denom = s * self.sigma * tau.sqrt();

    call_put_sign(option_type) * self.cash * disc * norm_pdf(d2) / denom
  }

  /// Gamma — $\partial^2 V/\partial S^2$.
  pub fn gamma(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    let (_, d2) = bsm_d1_d2(s, k, r - q, self.sigma, tau);
    let disc = (-r * tau).exp();
    let v = self.sigma;
    let sqrt_t = tau.sqrt();
    let pdf = norm_pdf(d2);

    -call_put_sign(option_type) * self.cash * disc * pdf * (1.0 + d2 * (v * sqrt_t))
      / (s * s * v * sqrt_t)
  }

  /// Vega — $\partial V/\partial \sigma$.
  pub fn vega(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    let (d1, d2) = bsm_d1_d2(s, k, r - q, self.sigma, tau);
    let disc = (-r * tau).exp();

    -call_put_sign(option_type) * self.cash * disc * norm_pdf(d2) * d1 / self.sigma
  }
}

impl ModelPricer for CashOrNothingPricer {
  /// `cash`/`sigma` are model state; `(s, k, r, q, tau)` is the query, with
  /// $b = r - q$ fed into $d_2$.
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let disc = (-r * tau).exp();
    if tau == 0.0 && self.sigma.is_finite() {
      let forward = s * ((r - q) * tau).exp();
      return self.cash * disc * expiry_indicator(OptionType::Call, forward, k);
    }
    let (_, d2) = bsm_d1_d2(s, k, r - q, self.sigma, tau);
    self.cash * disc * norm_cdf(d2)
  }

  /// The cash-or-nothing parity `C + P = Qe^{−rT}`, not the vanilla one (`cash_call_put_parity`).
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let disc = (-r * tau).exp();
    if tau == 0.0 && self.sigma.is_finite() {
      let forward = s * ((r - q) * tau).exp();
      return self.cash * disc * expiry_indicator(OptionType::Put, forward, k);
    }
    let (_, d2) = bsm_d1_d2(s, k, r - q, self.sigma, tau);
    self.cash * disc * norm_cdf(-d2)
  }
}

/// Asset-or-nothing digital: pays the underlying when the option finishes strictly in the money, so
/// nothing at `s = k`; at `tau = 0` it is worth that payoff.
///
/// $$
/// C_{\text{AoN}}=Se^{(b-r)T}N(d_1),\qquad P_{\text{AoN}}=Se^{(b-r)T}N(-d_1)
/// $$
///
/// ```
/// use stochastic_rs_quant::pricing::AssetOrNothingPricer;
/// use stochastic_rs_quant::traits::ModelPricer;
///
/// let model = AssetOrNothingPricer::new(0.25);
/// let itm = model.price_call(100.0, 90.0, 0.05, 0.02, 1.0);
/// let otm = model.price_call(100.0, 110.0, 0.05, 0.02, 1.0);
/// assert!(itm > otm);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AssetOrNothingPricer {
  /// Annualised Black–Scholes volatility.
  pub sigma: f64,
}

impl AssetOrNothingPricer {
  /// Panics for a negative or infinite `sigma`; a NaN is kept, as on [`CashOrNothingPricer::new`].
  pub fn new(sigma: f64) -> Self {
    assert_digital_sigma(sigma);
    Self { sigma }
  }

  /// Every Greek this pricer exposes at one query point, in a [`Greeks`] aggregate; the eight it
  /// does not expose stay [`f64::NAN`].
  pub fn greeks(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    option_type: OptionType,
  ) -> Greeks {
    if tau == 0.0 {
      return Greeks::nan();
    }
    Greeks {
      delta: self.delta(s, k, r, q, tau, option_type),
      ..Greeks::nan()
    }
  }

  /// Delta — $\partial V/\partial S$.
  pub fn delta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    let b = r - q;
    let d1 = bsm_d1(s, k, b, self.sigma, tau);
    let coc = ((b - r) * tau).exp();
    let v = self.sigma;
    let sqrt_t = tau.sqrt();
    let cdf_term = match option_type {
      OptionType::Call => norm_cdf(d1),
      OptionType::Put => norm_cdf(-d1),
    };

    coc * cdf_term + call_put_sign(option_type) * coc * norm_pdf(d1) / (v * sqrt_t)
  }
}

impl ModelPricer for AssetOrNothingPricer {
  /// `sigma` is model state; `(s, k, r, q, tau)` is the query, $b = r - q$.
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let b = r - q;
    let coc = ((b - r) * tau).exp();
    if tau == 0.0 && self.sigma.is_finite() {
      let forward = s * (b * tau).exp();
      return s * coc * expiry_indicator(OptionType::Call, forward, k);
    }
    let d1 = bsm_d1(s, k, b, self.sigma, tau);
    s * coc * norm_cdf(d1)
  }

  /// The asset-or-nothing parity `C + P = Se^{(b−r)T}`, not vanilla parity (`aon_call_put_parity`).
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let b = r - q;
    let coc = ((b - r) * tau).exp();
    if tau == 0.0 && self.sigma.is_finite() {
      let forward = s * (b * tau).exp();
      return s * coc * expiry_indicator(OptionType::Put, forward, k);
    }
    let d1 = bsm_d1(s, k, b, self.sigma, tau);
    s * coc * norm_cdf(-d1)
  }
}

/// Gap option, a vanilla when $K_1 = K_2$: pays $S - K_2$ when $S > K_1$ (call) or $K_2 - S$ when
/// $S < K_1$ (put), so nothing at $S = K_1$; at `tau = 0` it is worth that payoff.
///
/// $$
/// V = S e^{(b-r)T}N(d_1) - K_2 e^{-rT}N(d_2),\quad
/// d_1=\frac{\ln(S/K_1)+(b+\tfrac12\sigma^2)T}{\sigma\sqrt T}
/// $$
///
/// ```
/// use stochastic_rs_quant::pricing::GapPricer;
/// use stochastic_rs_quant::traits::ModelPricer;
///
/// // K2 = 100 on the model, K1 = 100 as the query strike: a vanilla call.
/// let model = GapPricer::new(100.0, 0.2);
/// let vanilla = model.price_call(100.0, 100.0, 0.05, 0.0, 1.0);
/// assert!((vanilla - 10.4506).abs() < 0.005);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GapPricer {
  /// Payoff strike `K₂`, a term of the contract; the query's `k` is the trigger `K₁` in `d₁`.
  pub k2: f64,
  /// Annualised Black–Scholes volatility.
  pub sigma: f64,
}

impl GapPricer {
  /// Panics for a negative or infinite `sigma` and an infinite `k2`; a NaN is kept, as on
  /// [`CashOrNothingPricer::new`]. The payoff strike `k2` is a contract term of either sign.
  pub fn new(k2: f64, sigma: f64) -> Self {
    assert_digital_sigma(sigma);
    assert!(
      !k2.is_infinite(),
      "k2 must satisfy `!k2.is_infinite()`, got k2 = {k2:?}"
    );
    Self { k2, sigma }
  }
}

impl ModelPricer for GapPricer {
  /// The query's `k` is the trigger `K₁` and `self.k2` the payoff strike `K₂`; at `k2 == k` a
  /// vanilla call (`gap_reduces_to_vanilla`).
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let b = r - q;
    let coc = ((b - r) * tau).exp();
    let disc = (-r * tau).exp();
    if tau == 0.0 && self.sigma.is_finite() {
      let pays = expiry_indicator(OptionType::Call, s, k);
      return s * coc * pays - self.k2 * disc * pays;
    }
    let (d1, d2) = bsm_d1_d2(s, k, b, self.sigma, tau);
    s * coc * norm_cdf(d1) - self.k2 * disc * norm_cdf(d2)
  }

  /// Overrides the trait's default with the gap put's own closed form
  /// rather than trusting vanilla parity against a single strike.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let b = r - q;
    let coc = ((b - r) * tau).exp();
    let disc = (-r * tau).exp();
    if tau == 0.0 && self.sigma.is_finite() {
      let pays = expiry_indicator(OptionType::Put, s, k);
      return self.k2 * disc * pays - s * coc * pays;
    }
    let (d1, d2) = bsm_d1_d2(s, k, b, self.sigma, tau);
    self.k2 * disc * norm_cdf(-d2) - s * coc * norm_cdf(-d1)
  }
}

/// Supershare option: pays $S_T / X_L$ when $X_L \le S_T \le X_H$, both band edges included; at
/// `tau = 0` it is worth that payoff.
///
/// $$
/// V = \frac{S}{X_L} e^{(b-r)T}[N(d_1) - N(d_2)]
/// $$
///
/// ```
/// use stochastic_rs_quant::pricing::SuperSharePricer;
/// use stochastic_rs_quant::traits::ModelPricer;
///
/// // X_H = 110 on the model, X_L = 90 as the query strike.
/// let model = SuperSharePricer::new(110.0, 0.2);
/// let v = model.price_call(100.0, 90.0, 0.05, 0.05, 0.25);
/// assert!(v > 0.0 && v < 100.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SuperSharePricer {
  /// Upper trigger `X_H`, a term of the contract; the query's `k` is `X_L`, the band edge the
  /// payoff is struck at and divided by.
  pub x_high: f64,
  /// Annualised Black–Scholes volatility.
  pub sigma: f64,
}

impl SuperSharePricer {
  /// Panics for a negative or infinite `sigma` (NaN stays, as on [`CashOrNothingPricer::new`]) and
  /// unless `x_high` is finite and `> 0`; a band inverted against the query's `k` prices negative.
  pub fn new(x_high: f64, sigma: f64) -> Self {
    assert_digital_sigma(sigma);
    assert!(
      x_high.is_finite() && x_high > 0.0,
      "x_high must satisfy `x_high.is_finite() && x_high > 0.0`, got x_high = {x_high:?}"
    );
    Self { x_high, sigma }
  }
}

impl ModelPricer for SuperSharePricer {
  /// The query's `k` is the lower trigger and payoff divisor `X_L`; `self.x_high` is `X_H`.
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let b = r - q;
    let coc = ((b - r) * tau).exp();
    if tau == 0.0 && self.sigma.is_finite() {
      let pays = if k <= s && s <= self.x_high { 1.0 } else { 0.0 };
      return s / k * coc * pays;
    }
    let d1 = bsm_d1(s, k, b, self.sigma, tau);
    let d2 = bsm_d1(s, self.x_high, b, self.sigma, tau);
    s / k * coc * (norm_cdf(d1) - norm_cdf(d2))
  }

  /// NaN: a supershare has no put (Haug 2007 defines the payoff above only), and the parity default
  /// would price an instrument that does not exist.
  fn price_put(&self, _s: f64, _k: f64, _r: f64, _q: f64, _tau: f64) -> f64 {
    f64::NAN
  }
}

/// `1` strictly past the strike, `0` at or short of it, as QuantLib's `CashOrNothingPayoff` and
/// `AssetOrNothingPayoff` pay; NaN for a NaN forward or strike.
pub(crate) fn expiry_indicator(option_type: OptionType, forward: f64, k: f64) -> f64 {
  if forward.is_nan() || k.is_nan() {
    return f64::NAN;
  }
  let in_the_money = match option_type {
    OptionType::Call => forward > k,
    OptionType::Put => forward < k,
  };
  if in_the_money { 1.0 } else { 0.0 }
}

/// `(s − k)⁺` for a call and `(k − s)⁺` for a put, NaN for a NaN `s` or `k`.
pub(crate) fn vanilla_payoff(option_type: OptionType, s: f64, k: f64) -> f64 {
  match option_type {
    OptionType::Call => (s - k).max_or_nan(0.0),
    OptionType::Put => (k - s).max_or_nan(0.0),
  }
}

/// The payoff at `tau = 0` under rates that are not NaN, NaN at or past expiry otherwise and for an
/// infinite `tau`, and `None` before expiry.
pub(crate) fn expired(tau: f64, rates: &[f64], payoff: impl FnOnce() -> f64) -> Option<f64> {
  if tau > 0.0 && tau < f64::INFINITY {
    return None;
  }
  let defined = tau == 0.0 && rates.iter().all(|rate| !rate.is_nan());
  Some(if defined { payoff() } else { f64::NAN })
}

/// Whether a spot or strike is zero, negative or NaN, where a lognormal formula has no value.
pub(crate) fn not_positive(values: &[f64]) -> bool {
  !values.iter().all(|&x| x > 0.0)
}

/// Whether a spot, strike or rate is infinite or NaN: with [`not_positive`], where a formula that
/// reads `ln s`, `ln k` and `e^{−rτ}` has no value.
pub(crate) fn not_finite(values: &[f64]) -> bool {
  !values.iter().all(|x| x.is_finite())
}

/// `+1` for a call and `−1` for a put: the put's `N(−d)` differentiates to `−φ(d)`.
const fn call_put_sign(option_type: OptionType) -> f64 {
  match option_type {
    OptionType::Call => 1.0,
    OptionType::Put => -1.0,
  }
}

/// A digital's volatility: not negative and not infinite; NaN is the unlinked handle's reading.
#[track_caller]
fn assert_digital_sigma(sigma: f64) {
  assert!(
    sigma.is_nan() || (sigma.is_finite() && sigma >= 0.0),
    "sigma must satisfy `sigma.is_nan() || (sigma.is_finite() && sigma >= 0.0)`, got sigma = {sigma:?}"
  );
}

/// $d_1=\frac{\ln(S/K)+(b+\sigma^2/2)T}{\sigma\sqrt T}$, the module's only copy: every price and
/// Greek above goes through it or [`bsm_d1_d2`]; NaN for a spot or strike that is not positive.
fn bsm_d1(s: f64, k: f64, b: f64, sigma: f64, t: f64) -> f64 {
  if not_positive(&[s, k]) {
    return f64::NAN;
  }
  ((s / k).ln() + (b + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt())
}

/// $(d_1,\ d_1-\sigma\sqrt T)$.
fn bsm_d1_d2(s: f64, k: f64, b: f64, sigma: f64, t: f64) -> (f64, f64) {
  let d1 = bsm_d1(s, k, b, sigma, t);
  (d1, d1 - sigma * t.sqrt())
}

#[cfg(test)]
#[path = "digital_tests.rs"]
mod tests;
