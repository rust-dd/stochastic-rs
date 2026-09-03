//! # Digital
//!
//! $$
//! C_{\text{CoN}}=Qe^{-rT}N(d_2),\quad C_{\text{AoN}}=Se^{(b-r)T}N(d_1),\quad
//! d_{1,2}=\frac{\ln(S/K)+(b\pm\tfrac12\sigma^2)T}{\sigma\sqrt T}
//! $$
//!
//! Every pricer here holds **model and contract state only** — the
//! volatility, plus whatever the payoff itself fixes (the cash payout, the
//! second strike, the upper band). Spot, strike, rate, dividend yield and
//! maturity are the pricing *query* and travel as arguments to
//! [`ModelPricer::price_call`] and to every Greek below, so one instance
//! prices a whole strike/maturity grid.
//!
//! The cost of carry $b$ is deliberately **not** a field. These four all use
//! $b = r - q$ (Merton's 1973 convention), recomputed from the query's own
//! rates on every call. A `b` fixed at construction would be a market
//! quantity frozen at one $(r, q)$ and then silently reused at another —
//! plausible in range, wrong in value, and invisible to any assertion on the
//! price alone.
//!
//! Source:
//! - Reiner, E. & Rubinstein, M. (1991), "Unscrambling the Binary Code"
//! - Haug, E. G. (2007), "The Complete Guide to Option Pricing Formulas", 2nd ed., Ch. 4
//! - Hull, J. (2018), "Options, Futures, and Other Derivatives", 10th ed., §26.9
//!
use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_distributions::special::norm_pdf;

use crate::OptionType;
use crate::traits::Greeks;
use crate::traits::ModelPricer;

/// Cash-or-nothing digital pays a fixed cash amount $Q$ when the option
/// finishes in the money.
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
  /// Cash payout $Q$ — a term of the contract, not a market quote, so it
  /// stays on the struct next to the volatility rather than travelling
  /// with the query.
  pub cash: f64,
  /// Volatility.
  pub sigma: f64,
}

impl CashOrNothingPricer {
  /// Validating constructor.
  ///
  /// # Panics
  /// - if `sigma` is negative. All four digitals share `bsm_d1`, so all
  ///   four share what a negative volatility does to it:
  ///   `1/(sigma*sqrt(tau))` flips sign, $d_1$ flips with it, and the price
  ///   that comes back is finite, plausible and wrong — negative for two of
  ///   the four. Case 1 of the crate's [failure
  ///   convention](crate::traits::ModelPricer#how-pricing-fails).
  ///
  /// A `NaN` `sigma` is deliberately **accepted**, unlike on
  /// [`HestonPricer::new`](crate::pricing::HestonPricer) where it is
  /// rejected. The difference is where the value comes from:
  /// `AnalyticBSEngine` builds these pricers with the volatility it reads
  /// off a market [`Handle`](crate::market::Handle), and an unlinked handle
  /// reads as `NaN` *by design* — the crate's missing-data answer, the same
  /// one [`TimeExt::tau_or_from_dates`](crate::traits::TimeExt) gives. That
  /// is case 2, and it has to propagate into a `NaN` NPV rather than abort
  /// the engine; `every_unlinked_handle_poisons_npv_and_greeks` pins it.
  /// The check spells the missing-data case out, `sigma.is_nan() || sigma >= 0.0`,
  /// precisely so the two cases stay apart and neither is a side effect of
  /// how `NaN` compares. Heston's parameters have no such path — they arrive
  /// from calibration output, never from a handle.
  ///
  /// `sigma == 0` is the deterministic limit and stays accepted.
  ///
  /// `cash` is a contract term with no invalid value — a negative payout is
  /// a short digital — so it is deliberately unchecked.
  pub fn new(cash: f64, sigma: f64) -> Self {
    assert!(
      sigma.is_nan() || sigma >= 0.0,
      "CashOrNothingPricer::new: sigma must be a non-negative volatility (got {sigma})"
    );
    Self { cash, sigma }
  }

  /// Every Greek this pricer exposes at one query point, in a [`Greeks`]
  /// aggregate; the six it does not expose stay [`f64::NAN`].
  ///
  /// This is what the removed
  /// [`GreeksExt`](crate::traits::GreeksExt) impl's `greeks()` provided.
  /// The trait's accessors take no arguments, so only a type that already
  /// carries a query can implement it, and this one no longer does —
  /// `BSMPricer` and `HestonPricer` came off the trait the same way.
  /// Callers that want the whole set go through here rather than
  /// hand-assembling a nine-field struct literal, which is where a
  /// mis-mapped member loses its only pin — see
  /// `digital_greeks_aggregates_match_their_accessors`.
  pub fn greeks(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    option_type: OptionType,
  ) -> Greeks {
    Greeks {
      delta: self.delta(s, k, r, q, tau, option_type),
      gamma: self.gamma(s, k, r, q, tau, option_type),
      vega: self.vega(s, k, r, q, tau, option_type),
      ..Greeks::nan()
    }
  }

  /// Delta — $\partial V/\partial S$.
  pub fn delta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    let (_, d2) = bsm_d1_d2(s, k, r - q, self.sigma, tau);
    let disc = (-r * tau).exp();
    let denom = s * self.sigma * tau.sqrt();

    call_put_sign(option_type) * self.cash * disc * norm_pdf(d2) / denom
  }

  /// Gamma — $\partial^2 V/\partial S^2$.
  pub fn gamma(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
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
    let (d1, d2) = bsm_d1_d2(s, k, r - q, self.sigma, tau);
    let disc = (-r * tau).exp();

    -call_put_sign(option_type) * self.cash * disc * norm_pdf(d2) * d1 / self.sigma
  }
}

impl ModelPricer for CashOrNothingPricer {
  /// `cash`/`sigma` are model state; `(s, k, r, q, tau)` is the query, with
  /// $b = r - q$ fed into $d_2$.
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let (_, d2) = bsm_d1_d2(s, k, r - q, self.sigma, tau);
    let disc = (-r * tau).exp();
    self.cash * disc * norm_cdf(d2)
  }

  /// Overrides the trait's vanilla-parity default: here
  /// $C_{\text{CoN}} + P_{\text{CoN}} = Qe^{-rT}$, not
  /// $C - Se^{-qT} + Ke^{-rT}$ — see `cash_call_put_parity`.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let (_, d2) = bsm_d1_d2(s, k, r - q, self.sigma, tau);
    let disc = (-r * tau).exp();
    self.cash * disc * norm_cdf(-d2)
  }
}

/// Asset-or-nothing digital pays the underlying when the option finishes in
/// the money.
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
  /// Volatility.
  pub sigma: f64,
}

impl AssetOrNothingPricer {
  /// Validating constructor.
  ///
  /// # Panics
  /// - if `sigma` is negative. All four digitals share `bsm_d1`, so all
  ///   four share what a negative volatility does to it:
  ///   `1/(sigma*sqrt(tau))` flips sign, $d_1$ flips with it, and the price
  ///   that comes back is finite, plausible and wrong — negative for two of
  ///   the four. Case 1 of the crate's [failure
  ///   convention](crate::traits::ModelPricer#how-pricing-fails).
  ///
  /// A `NaN` `sigma` is deliberately **accepted**, unlike on
  /// [`HestonPricer::new`](crate::pricing::HestonPricer) where it is
  /// rejected. The difference is where the value comes from:
  /// `AnalyticBSEngine` builds these pricers with the volatility it reads
  /// off a market [`Handle`](crate::market::Handle), and an unlinked handle
  /// reads as `NaN` *by design* — the crate's missing-data answer, the same
  /// one [`TimeExt::tau_or_from_dates`](crate::traits::TimeExt) gives. That
  /// is case 2, and it has to propagate into a `NaN` NPV rather than abort
  /// the engine; `every_unlinked_handle_poisons_npv_and_greeks` pins it.
  /// The check spells the missing-data case out, `sigma.is_nan() || sigma >= 0.0`,
  /// precisely so the two cases stay apart and neither is a side effect of
  /// how `NaN` compares. Heston's parameters have no such path — they arrive
  /// from calibration output, never from a handle.
  ///
  /// `sigma == 0` is the deterministic limit and stays accepted.
  pub fn new(sigma: f64) -> Self {
    assert!(
      sigma.is_nan() || sigma >= 0.0,
      "AssetOrNothingPricer::new: sigma must be a non-negative volatility (got {sigma})"
    );
    Self { sigma }
  }

  /// Every Greek this pricer exposes at one query point; the eight it does
  /// not expose stay [`f64::NAN`]. See
  /// [`CashOrNothingPricer::greeks`] for why this is an inherent method
  /// rather than a [`GreeksExt`](crate::traits::GreeksExt) impl.
  pub fn greeks(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    option_type: OptionType,
  ) -> Greeks {
    Greeks {
      delta: self.delta(s, k, r, q, tau, option_type),
      ..Greeks::nan()
    }
  }

  /// Delta — $\partial V/\partial S$.
  pub fn delta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
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
    let d1 = bsm_d1(s, k, b, self.sigma, tau);
    let coc = ((b - r) * tau).exp();
    s * coc * norm_cdf(d1)
  }

  /// Overrides the trait's vanilla-parity default: here
  /// $C_{\text{AoN}} + P_{\text{AoN}} = Se^{(b-r)T}$ — see
  /// `aon_call_put_parity` — not vanilla put-call parity.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let b = r - q;
    let d1 = bsm_d1(s, k, b, self.sigma, tau);
    let coc = ((b - r) * tau).exp();
    s * coc * norm_cdf(-d1)
  }
}

/// Gap option: pays $S - K_2$ when $S > K_1$ (call) or $K_2 - S$ when
/// $S < K_1$ (put). Reduces to a vanilla when $K_1 = K_2$.
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
  /// Payoff strike $K_2$ — a term of the contract. The *trigger* strike
  /// $K_1$ is the query's `k`, since it is $K_1$ that sets the moneyness
  /// boundary and so enters $d_1$.
  pub k2: f64,
  /// Volatility.
  pub sigma: f64,
}

impl GapPricer {
  /// Validating constructor.
  ///
  /// # Panics
  /// - if `sigma` is negative. All four digitals share `bsm_d1`, so all
  ///   four share what a negative volatility does to it:
  ///   `1/(sigma*sqrt(tau))` flips sign, $d_1$ flips with it, and the price
  ///   that comes back is finite, plausible and wrong — negative for two of
  ///   the four. Case 1 of the crate's [failure
  ///   convention](crate::traits::ModelPricer#how-pricing-fails).
  ///
  /// A `NaN` `sigma` is deliberately **accepted**, unlike on
  /// [`HestonPricer::new`](crate::pricing::HestonPricer) where it is
  /// rejected. The difference is where the value comes from:
  /// `AnalyticBSEngine` builds these pricers with the volatility it reads
  /// off a market [`Handle`](crate::market::Handle), and an unlinked handle
  /// reads as `NaN` *by design* — the crate's missing-data answer, the same
  /// one [`TimeExt::tau_or_from_dates`](crate::traits::TimeExt) gives. That
  /// is case 2, and it has to propagate into a `NaN` NPV rather than abort
  /// the engine; `every_unlinked_handle_poisons_npv_and_greeks` pins it.
  /// The check spells the missing-data case out, `sigma.is_nan() || sigma >= 0.0`,
  /// precisely so the two cases stay apart and neither is a side effect of
  /// how `NaN` compares. Heston's parameters have no such path — they arrive
  /// from calibration output, never from a handle.
  ///
  /// `sigma == 0` is the deterministic limit and stays accepted.
  ///
  /// `k2` is the payoff strike — a contract term, not a market or model
  /// quantity — so like [`CashOrNothingPricer`]'s `cash` it has no invalid
  /// value and is unchecked.
  pub fn new(k2: f64, sigma: f64) -> Self {
    assert!(
      sigma.is_nan() || sigma >= 0.0,
      "GapPricer::new: sigma must be a non-negative volatility (got {sigma})"
    );
    Self { k2, sigma }
  }
}

impl ModelPricer for GapPricer {
  /// $K_1$ (the moneyness trigger) is the query strike `k`; $K_2$ (the
  /// payoff strike) and `sigma` stay model state on `self.k2`/`self.sigma`
  /// — reduces to a vanilla call when `self.k2 == k`, see
  /// `gap_reduces_to_vanilla`.
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let b = r - q;
    let (d1, d2) = bsm_d1_d2(s, k, b, self.sigma, tau);
    let coc = ((b - r) * tau).exp();
    let disc = (-r * tau).exp();
    s * coc * norm_cdf(d1) - self.k2 * disc * norm_cdf(d2)
  }

  /// Overrides the trait's default with the gap put's own closed form
  /// rather than trusting vanilla parity against a single strike.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let b = r - q;
    let (d1, d2) = bsm_d1_d2(s, k, b, self.sigma, tau);
    let coc = ((b - r) * tau).exp();
    let disc = (-r * tau).exp();
    self.k2 * disc * norm_cdf(-d2) - s * coc * norm_cdf(-d1)
  }
}

/// Supershare option pays $S_T / X_L$ when $X_L \le S_T \le X_H$.
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
  /// Upper trigger $X_H$ — a term of the contract. The *lower* trigger
  /// $X_L$ is the query's `k`: it is the band edge the payoff is struck
  /// and divided by, so it is the one of the pair that behaves like a
  /// strike.
  pub x_high: f64,
  /// Volatility.
  pub sigma: f64,
}

impl SuperSharePricer {
  /// Validating constructor.
  ///
  /// # Panics
  /// - if `sigma` is negative. All four digitals share `bsm_d1`, so all
  ///   four share what a negative volatility does to it:
  ///   `1/(sigma*sqrt(tau))` flips sign, $d_1$ flips with it, and the price
  ///   that comes back is finite, plausible and wrong — negative for two of
  ///   the four. Case 1 of the crate's [failure
  ///   convention](crate::traits::ModelPricer#how-pricing-fails).
  ///
  /// A `NaN` `sigma` is deliberately **accepted**, unlike on
  /// [`HestonPricer::new`](crate::pricing::HestonPricer) where it is
  /// rejected. The difference is where the value comes from:
  /// `AnalyticBSEngine` builds these pricers with the volatility it reads
  /// off a market [`Handle`](crate::market::Handle), and an unlinked handle
  /// reads as `NaN` *by design* — the crate's missing-data answer, the same
  /// one [`TimeExt::tau_or_from_dates`](crate::traits::TimeExt) gives. That
  /// is case 2, and it has to propagate into a `NaN` NPV rather than abort
  /// the engine; `every_unlinked_handle_poisons_npv_and_greeks` pins it.
  /// The check spells the missing-data case out, `sigma.is_nan() || sigma >= 0.0`,
  /// precisely so the two cases stay apart and neither is a side effect of
  /// how `NaN` compares. Heston's parameters have no such path — they arrive
  /// from calibration output, never from a handle.
  ///
  /// `sigma == 0` is the deterministic limit and stays accepted.
  /// - if `x_high` is not strictly positive, or `NaN` — the upper band edge
  ///   is a price level, and a non-positive one is not a band.
  ///
  /// `x_high` is a contract term the caller states outright, with no
  /// handle behind it, so unlike `sigma` it has no missing-data reading and
  /// a `NaN` is rejected along with the non-positive values.
  ///
  /// An *inverted* band (`x_high` below the query's `k`) is deliberately
  /// **not** rejected here even though it prices negative: `k` is the query
  /// strike, so no single argument to this constructor is invalid in that
  /// case — it is a model/query combination, which this layer cannot see.
  pub fn new(x_high: f64, sigma: f64) -> Self {
    assert!(
      sigma.is_nan() || sigma >= 0.0,
      "SuperSharePricer::new: sigma must be a non-negative volatility (got {sigma})"
    );
    assert!(
      x_high > 0.0,
      "SuperSharePricer::new: x_high must be strictly positive (got {x_high})"
    );
    Self { x_high, sigma }
  }
}

impl ModelPricer for SuperSharePricer {
  /// $X_L$ (the lower trigger, also the payoff divisor) is the query
  /// strike `k`; $X_H$ and `sigma` stay model state on `self.x_high`/
  /// `self.sigma`.
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let b = r - q;
    let d1 = bsm_d1(s, k, b, self.sigma, tau);
    let d2 = bsm_d1(s, self.x_high, b, self.sigma, tau);
    let coc = ((b - r) * tau).exp();
    s / k * coc * (norm_cdf(d1) - norm_cdf(d2))
  }

  /// A supershare has no put analogue — Haug (2007) defines only the
  /// payoff above — so this returns `NaN` rather than the trait's
  /// vanilla-parity default, which would silently return a number with no
  /// corresponding instrument.
  fn price_put(&self, _s: f64, _k: f64, _r: f64, _q: f64, _tau: f64) -> f64 {
    f64::NAN
  }
}

/// $+1$ for a call, $-1$ for a put — the sign a digital Greek picks up
/// because the put's $N(-d)$ differentiates to $-\phi(d)$ where the call's
/// $N(d)$ gives $+\phi(d)$.
const fn call_put_sign(option_type: OptionType) -> f64 {
  match option_type {
    OptionType::Call => 1.0,
    OptionType::Put => -1.0,
  }
}

/// $d_1=\frac{\ln(S/K)+(b+\sigma^2/2)T}{\sigma\sqrt T}$ — the standardized
/// moneyness term, and the **only** copy of it in this module. Every price
/// and every Greek above routes through here or through
/// [`bsm_d1_d2`]; the four per-struct `d1_d2(&self)` methods that used to
/// sit alongside it read the bundled query fields, so removing those fields
/// left this the single source.
fn bsm_d1(s: f64, k: f64, b: f64, sigma: f64, t: f64) -> f64 {
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
