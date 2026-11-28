use stochastic_rs_distributions::special::norm_cdf;

use crate::CallPut;
use crate::OptionType;
use crate::pricing::digital::assert_handle_sigma;
use crate::pricing::digital::vanilla_payoff;
use crate::traits::ModelPricer;
use crate::traits::VanillaEuropeanCall;

#[derive(Default, Debug, Clone, Copy)]
#[non_exhaustive]
pub enum BsmCoc {
  /// Black-Scholes-Merton 1973 (stock option)
  /// Cost of carry = risk-free rate
  #[default]
  Bsm1973,
  /// Black-Scholes-Merton 1976 (stock option)
  /// Cost of carry = risk-free rate - dividend yield
  Merton1973,
  /// Black 1976 (futures option)
  /// Cost of carry = 0
  Black1976,
  /// Asay 1982 (futures option)
  /// Cost of carry = 0
  Asay1982,
  /// Garman-Kohlhagen 1983 (currency option)
  /// Cost of carry = (domestic - foregin) risk-free rate
  GarmanKohlhagen1983,
}

/// Black–Scholes–Merton with a generalised cost of carry, holding the volatility and the [`BsmCoc`]
/// convention; the query travels as arguments to every price and Greek.
///
/// ```
/// use stochastic_rs_quant::pricing::bsm::{BsmCoc, BsmPricer};
/// use stochastic_rs_quant::traits::ModelPricer;
///
/// let model = BsmPricer::new(0.2, BsmCoc::Merton1973);
/// let atm = model.price_call(100.0, 100.0, 0.05, 0.0, 1.0);
/// let otm = model.price_call(100.0, 120.0, 0.05, 0.0, 1.0);
/// assert!(atm > otm);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BsmPricer {
  /// Volatility
  pub sigma: f64,
  /// Cost-of-carry convention
  pub b: BsmCoc,
}

impl BsmPricer {
  /// Panics for a negative or infinite `sigma`, a negative one flipping `d₁` into a wrong price; a
  /// NaN is kept, as `AnalyticBsEngine` reads one off an unlinked handle and must price NaN.
  pub fn new(sigma: f64, b: BsmCoc) -> Self {
    assert_handle_sigma(sigma);
    Self { sigma, b }
  }

  /// The carry `b` at the query's rates, `(r, q)` domestic and foreign under Garman–Kohlhagen;
  /// [`BsmCoc::Asay1982`] is Black 1976's `b = 0` and `e^{−rτ}` discount, margining unmodelled.
  pub fn b(&self, r: f64, q: f64) -> f64 {
    match self.b {
      BsmCoc::Bsm1973 => r,
      BsmCoc::Merton1973 | BsmCoc::GarmanKohlhagen1983 => r - q,
      BsmCoc::Black1976 | BsmCoc::Asay1982 => 0.0,
    }
  }

  /// $d_1$, $d_2$ at one query point; NaN for a spot or strike that is not positive.
  pub fn d1_d2(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> (f64, f64) {
    if s.is_nan() || s <= 0.0 || k.is_nan() || k <= 0.0 {
      return (f64::NAN, f64::NAN);
    }
    let d1 = (1.0 / (self.sigma * tau.sqrt()))
      * ((s / k).ln() + (self.b(r, q) + 0.5 * self.sigma.powi(2)) * tau);
    let d2 = d1 - self.sigma * tau.sqrt();

    (d1, d2)
  }

  /// The carried spot `S e^{(b−r)τ}`, the discounted strike and `(d₁, d₂)`, which are `None` at
  /// `tau = 0` under a finite `sigma`, where a leg is its payoff.
  fn terms(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> (f64, f64, Option<(f64, f64)>) {
    let carry = ((self.b(r, q) - r) * tau).exp();
    let disc = (-r * tau).exp();
    let at_expiry = tau == 0.0 && self.sigma.is_finite();
    let d = (!at_expiry).then(|| self.d1_d2(s, k, r, q, tau));
    (s * carry, k * disc, d)
  }

  /// One leg from [`terms`](Self::terms): at `tau = 0` the payoff, NaN for a non-finite `sigma`.
  ///
  /// $$
  /// C=Se^{(b-r)\tau}N(d_1)-Ke^{-r\tau}N(d_2),\qquad
  /// P=-Se^{(b-r)\tau}N(-d_1)+Ke^{-r\tau}N(-d_2)
  /// $$
  fn leg((spot, strike, d): (f64, f64, Option<(f64, f64)>), option_type: OptionType) -> f64 {
    match (d, option_type) {
      (None, _) => vanilla_payoff(option_type, spot, strike),
      (Some((d1, d2)), OptionType::Call) => spot * norm_cdf(d1) - strike * norm_cdf(d2),
      (Some((d1, d2)), OptionType::Put) => -spot * norm_cdf(-d1) + strike * norm_cdf(-d2),
    }
  }

  /// Both legs off one `(d₁, d₂)`, each bit for bit what `price_call` and `price_put` answer.
  pub(crate) fn price_call_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> CallPut {
    let terms = self.terms(s, k, r, q, tau);
    CallPut {
      call: Self::leg(terms, OptionType::Call),
      put: Self::leg(terms, OptionType::Put),
    }
  }

  /// Black volatility implied by `c_price` under this convention's carry [`b`](Self::b), whatever
  /// `sigma` is; NaN for a price outside the no-arbitrage bounds.
  // One argument per term of the query.
  #[allow(clippy::too_many_arguments)]
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
    let forward = s * (self.b(r, q) * tau).exp();
    let undiscounted_price = c_price * (r * tau).exp();
    crate::vol_surface::implied::black_implied_vol(undiscounted_price, forward, k, tau, option_type)
  }
}

impl ModelPricer for BsmPricer {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    Self::leg(self.terms(s, k, r, q, tau), OptionType::Call)
  }

  /// Parity on this carry, `C − P = Se^{(b−r)τ} − Ke^{−rτ}`, which the default `e^{−qτ}` matches
  /// only where `b = r − q` (`bsm_price_put_overrides_vanilla_parity`).
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    Self::leg(self.terms(s, k, r, q, tau), OptionType::Put)
  }
}

/// European vanilla call — but **not** always on $Se^{(r-q)\tau}$, which is
/// why this overrides the default forward.
impl VanillaEuropeanCall for BsmPricer {
  /// `Se^{bτ}` at [`BsmPricer::b`]; the default `Se^{(r−q)τ}` would invert a `Bsm1973` surface at
  /// `q ≠ 0`, or a Black 1976 or Asay one at `q ≠ r`, into a smile (the `carry_surface` tests).
  fn vanilla_call_forward(&self, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    s * (self.b(r, q) * tau).exp()
  }
}
