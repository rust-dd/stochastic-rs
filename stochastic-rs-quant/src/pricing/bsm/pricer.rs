use implied_vol::DefaultSpecialFn;
use implied_vol::ImpliedBlackVolatility;
use stochastic_rs_distributions::special::norm_cdf;

use crate::OptionType;
use crate::traits::ModelPricer;
use crate::traits::VanillaEuropeanCall;

#[derive(Default, Debug, Clone, Copy)]
#[non_exhaustive]
pub enum BSMCoc {
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
/// use stochastic_rs_quant::pricing::bsm::{BSMCoc, BSMPricer};
/// use stochastic_rs_quant::traits::ModelPricer;
///
/// let model = BSMPricer::new(0.2, BSMCoc::Merton1973);
/// let atm = model.price_call(100.0, 100.0, 0.05, 0.0, 1.0);
/// let otm = model.price_call(100.0, 120.0, 0.05, 0.0, 1.0);
/// assert!(atm > otm);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BSMPricer {
  /// Volatility
  pub v: f64,
  /// Cost-of-carry convention
  pub b: BSMCoc,
}

impl BSMPricer {
  pub const fn new(v: f64, b: BSMCoc) -> Self {
    Self { v, b }
  }

  /// Cost of carry $b$ implied by the convention at the query's rates.
  ///
  /// The query's `(r, q)` pair carries both roles the pre-query struct
  /// spread across four rate fields: under
  /// [`BSMCoc::GarmanKohlhagen1983`] they are the *domestic* and *foreign*
  /// rates, so $b = r_d - r_f$ is Garman-Kohlhagen's conventional
  /// Black-Scholes embedding (foreign rate in the dividend slot) rather
  /// than an approximation of it.
  ///
  /// [`BSMCoc::Asay1982`] returns the same `0.0` as
  /// [`BSMCoc::Black1976`] and discounts at the same `exp(-r * tau)`;
  /// margined-futures' zero discounting is *not* modelled here and never
  /// was.
  pub fn b(&self, r: f64, q: f64) -> f64 {
    match self.b {
      BSMCoc::Bsm1973 => r,
      BSMCoc::Merton1973 | BSMCoc::GarmanKohlhagen1983 => r - q,
      BSMCoc::Black1976 | BSMCoc::Asay1982 => 0.0,
    }
  }

  /// $d_1$, $d_2$ at one query point.
  pub fn d1_d2(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> (f64, f64) {
    let d1 =
      (1.0 / (self.v * tau.sqrt())) * ((s / k).ln() + (self.b(r, q) + 0.5 * self.v.powi(2)) * tau);
    let d2 = d1 - self.v * tau.sqrt();

    (d1, d2)
  }

  /// Call and put price at one query point.
  ///
  /// $$
  /// C=Se^{(b-r)\tau}N(d_1)-Ke^{-r\tau}N(d_2),\qquad
  /// P=-Se^{(b-r)\tau}N(-d_1)+Ke^{-r\tau}N(-d_2)
  /// $$
  pub fn call_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> (f64, f64) {
    let (d1, d2) = self.d1_d2(s, k, r, q, tau);
    let carry = ((self.b(r, q) - r) * tau).exp();
    let disc = (-r * tau).exp();

    let call = s * carry * norm_cdf(d1) - k * disc * norm_cdf(d2);
    let put = -s * carry * norm_cdf(-d1) + k * disc * norm_cdf(-d2);

    (call, put)
  }

  /// Black volatility implied by `price` at one query point.
  ///
  /// Depends on the cost-of-carry convention ([`b`](Self::b)) but *not* on
  /// [`v`](Self::v) — it inverts the price for a volatility rather than
  /// pricing at one, so any instance sharing the convention returns the
  /// same answer.
  ///
  /// Returns [`f64::NAN`] when the price is outside the no-arbitrage bounds
  /// the inversion can invert.
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

impl ModelPricer for BSMPricer {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put(s, k, r, q, tau).0
  }

  /// Overrides the trait's vanilla-parity default, which assumes the carry
  /// factor is $e^{-q\tau}$. Here it is $e^{(b-r)\tau}$, so parity reads
  /// $C-P=Se^{(b-r)\tau}-Ke^{-r\tau}$ and the two agree only when
  /// $b=r-q$ — true for [`BSMCoc::Merton1973`] and
  /// [`BSMCoc::GarmanKohlhagen1983`], false for [`BSMCoc::Bsm1973`] at
  /// `q != 0` and for [`BSMCoc::Black1976`] / [`BSMCoc::Asay1982`] at
  /// `q != r`. See `bsm_price_put_overrides_vanilla_parity`.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put(s, k, r, q, tau).1
  }
}

/// European vanilla call — but **not** always on $Se^{(r-q)\tau}$, which is
/// why this overrides the default forward.
impl VanillaEuropeanCall for BSMPricer {
  /// $Se^{b\tau}$ at this pricer's own cost of carry, which is
  /// [`b(r, q)`](BSMPricer::b) and equals $r - q$ for only two of the five
  /// [`BSMCoc`] conventions.
  ///
  /// The default $Se^{(r-q)\tau}$ would be the *futures price grown at the
  /// risk-free rate* under [`BSMCoc::Black1976`] / [`BSMCoc::Asay1982`],
  /// where $b = 0$ and the forward is just $S$; under [`BSMCoc::Bsm1973`] at
  /// `q != 0` it would discount a dividend the price never paid. Both leave
  /// every price inside the no-arbitrage band, so the surface inverts them
  /// into a full smile — see
  /// `futures_carry_surface_recovers_the_models_flat_vol` and
  /// `dividend_ignoring_carry_surface_recovers_the_models_flat_vol`, both of
  /// which fail against the default. [`BSMCoc::Merton1973`] and
  /// [`BSMCoc::GarmanKohlhagen1983`] have $b = r - q$ and so land on the
  /// default's exact expression, bit for bit.
  ///
  /// [`BSMCoc`]: crate::pricing::bsm::BSMCoc
  fn vanilla_call_forward(&self, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    s * (self.b(r, q) * tau).exp()
  }
}
