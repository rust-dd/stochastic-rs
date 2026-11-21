use stochastic_rs_distributions::special::norm_cdf;

use crate::CallPut;
use crate::OptionType;
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

/// Black-Scholes-Merton generalised-cost-of-carry model.
///
/// The struct holds **model state only** — the volatility and the
/// cost-of-carry convention. Spot, strike, rate, dividend yield and
/// maturity are the pricing *query* and travel as arguments to
/// [`ModelPricer::price_call`] and to every Greek below, so one instance
/// prices a whole strike/maturity grid.
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
  pub const fn new(sigma: f64, b: BsmCoc) -> Self {
    Self { sigma, b }
  }

  /// Cost of carry $b$ implied by the convention at the query's rates.
  ///
  /// The query's `(r, q)` pair carries both roles the pre-query struct
  /// spread across four rate fields: under
  /// [`BsmCoc::GarmanKohlhagen1983`] they are the *domestic* and *foreign*
  /// rates, so $b = r_d - r_f$ is Garman-Kohlhagen's conventional
  /// Black-Scholes embedding (foreign rate in the dividend slot) rather
  /// than an approximation of it.
  ///
  /// [`BsmCoc::Asay1982`] returns the same `0.0` as
  /// [`BsmCoc::Black1976`] and discounts at the same `exp(-r * tau)`;
  /// margined-futures' zero discounting is *not* modelled here and never
  /// was.
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

  /// Overrides the trait's vanilla-parity default, which assumes the carry
  /// factor is $e^{-q\tau}$. Here it is $e^{(b-r)\tau}$, so parity reads
  /// $C-P=Se^{(b-r)\tau}-Ke^{-r\tau}$ and the two agree only when
  /// $b=r-q$ — true for [`BsmCoc::Merton1973`] and
  /// [`BsmCoc::GarmanKohlhagen1983`], false for [`BsmCoc::Bsm1973`] at
  /// `q != 0` and for [`BsmCoc::Black1976`] / [`BsmCoc::Asay1982`] at
  /// `q != r`. See `bsm_price_put_overrides_vanilla_parity`.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    Self::leg(self.terms(s, k, r, q, tau), OptionType::Put)
  }
}

/// European vanilla call — but **not** always on $Se^{(r-q)\tau}$, which is
/// why this overrides the default forward.
impl VanillaEuropeanCall for BsmPricer {
  /// $Se^{b\tau}$ at this pricer's own cost of carry, which is
  /// [`b(r, q)`](BsmPricer::b) and equals $r - q$ for only two of the five
  /// [`BsmCoc`] conventions.
  ///
  /// The default $Se^{(r-q)\tau}$ would be the *futures price grown at the
  /// risk-free rate* under [`BsmCoc::Black1976`] / [`BsmCoc::Asay1982`],
  /// where $b = 0$ and the forward is just $S$; under [`BsmCoc::Bsm1973`] at
  /// `q != 0` it would discount a dividend the price never paid. Both leave
  /// every price inside the no-arbitrage band, so the surface inverts them
  /// into a full smile — see
  /// `futures_carry_surface_recovers_the_models_flat_vol` and
  /// `dividend_ignoring_carry_surface_recovers_the_models_flat_vol`, both of
  /// which fail against the default. [`BsmCoc::Merton1973`] and
  /// [`BsmCoc::GarmanKohlhagen1983`] have $b = r - q$ and so land on the
  /// default's exact expression, bit for bit.
  ///
  /// [`BsmCoc`]: crate::pricing::bsm::BsmCoc
  fn vanilla_call_forward(&self, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    s * (self.b(r, q) * tau).exp()
  }
}
