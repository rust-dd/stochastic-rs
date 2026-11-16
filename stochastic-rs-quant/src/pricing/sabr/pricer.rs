use implied_vol::DefaultSpecialFn;
use implied_vol::ImpliedBlackVolatility;

use crate::OptionType;
use crate::pricing::bsm::BsmCoc;
use crate::pricing::bsm::BsmPricer;
use crate::pricing::sabr::hagan::forward_fx;
use crate::pricing::sabr::hagan::fx_delta_from_forward;
use crate::pricing::sabr::hagan::hagan_implied_vol;
use crate::pricing::sabr::params::SabrParams;
use crate::traits::ModelPricer;
use crate::traits::VanillaEuropeanCall;

/// Hagan (2002) general-β SABR: its implied vol through Black–Scholes at carry `b = r − q`, which
/// is Garman–Kohlhagen under the FX reading `(r, q) = (r_d, r_f)`.
///
/// ```
/// use stochastic_rs_quant::pricing::sabr::SabrParams;
/// use stochastic_rs_quant::pricing::sabr::SabrPricer;
/// use stochastic_rs_quant::traits::ModelPricer;
///
/// let model = SabrPricer::new(SabrParams { alpha: 0.2, beta: 1.0, rho: -0.3, nu: 0.4 });
/// let atm = model.price_call(100.0, 100.0, 0.05, 0.0, 1.0);
/// let otm = model.price_call(100.0, 130.0, 0.05, 0.0, 1.0);
/// assert!(atm > otm);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SabrPricer {
  params: SabrParams,
}

impl SabrPricer {
  /// Panics unless `params` passes [`SabrParams::new`], whose checks hold everywhere inside
  /// [`SabrCalibrator`](crate::calibration::sabr::SabrCalibrator)'s projection box.
  pub fn new(params: SabrParams) -> Self {
    let SabrParams {
      alpha,
      beta,
      rho,
      nu,
    } = params;
    Self {
      params: SabrParams::new(alpha, beta, rho, nu),
    }
  }

  pub fn params(&self) -> SabrParams {
    self.params
  }

  /// Forward at one query point, `s·e^{(r-q)τ}`. Under the FX reading
  /// `(r, q)` are the domestic and foreign rates.
  pub fn forward(&self, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    forward_fx(s, tau, r, q)
  }

  /// Hagan vol at `k` against [`forward`](Self::forward); NaN for a query outside the domain, and
  /// possibly non-positive on legal parameters (see [`call_put`](Self::call_put)).
  pub fn sigma(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    hagan_implied_vol(k, self.forward(s, r, q, tau), tau, &self.params)
  }

  /// Forward-based (premium-included) FX delta, with the foreign rate read
  /// off the query's `q` slot.
  pub fn sabr_fx_forward_delta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, phi: f64) -> f64 {
    fx_delta_from_forward(
      k,
      self.forward(s, r, q, tau),
      self.sigma(s, k, r, q, tau),
      tau,
      q,
      phi,
    )
  }

  /// Call and put at one query point, both NaN unless [`sigma`](Self::sigma) is finite and > 0,
  /// as where Hagan's small-τ bracket turns negative (`α 0.2, β 1, ρ −0.9, ν 3` at `τ = 10`).
  pub fn call_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> (f64, f64) {
    let sigma = self.sigma(s, k, r, q, tau);
    if !sigma.is_finite() || sigma <= 0.0 {
      return (f64::NAN, f64::NAN);
    }
    BsmPricer::new(sigma, BsmCoc::Merton1973).call_put(s, k, r, q, tau)
  }

  /// Black volatility implied by `price` at one query point.
  ///
  /// Depends on none of the four Sabr parameters — it inverts a price for a
  /// volatility rather than pricing at one, so any `SabrPricer` returns the
  /// same answer. Kept as an inherent method on this type (rather than a
  /// free function) because it is the inverse of
  /// [`call_put`](Self::call_put) and shares its `b = r - q` carry
  /// convention.
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
    let forward = self.forward(s, r, q, tau);
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

impl ModelPricer for SabrPricer {
  /// NaN outside the domain or at a degenerate Hagan vol; see [`call_put`](SabrPricer::call_put).
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put(s, k, r, q, tau).0
  }

  /// Takes the Black-Scholes closed-form put rather than the trait's
  /// vanilla-parity default. The two are *mathematically* the same here —
  /// the carry is `b = r - q`, which is exactly the case where vanilla
  /// parity holds — but the closed form is what the pre-query
  /// `calculate_call_put().1` returned, so delegating keeps the number
  /// bit-identical rather than merely equal to within rounding. See
  /// `sabr_price_put_matches_parity_but_is_the_closed_form`.
  ///
  /// Panics and returns [`f64::NAN`] under exactly the same conditions as
  /// [`price_call`](SabrPricer::price_call), since both read the same
  /// [`call_put`](SabrPricer::call_put) pair.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put(s, k, r, q, tau).1
  }
}

/// European vanilla call: Hagan's expansion produces a Black volatility
/// that [`call_put`](SabrPricer::call_put) prices as a vanilla against
/// [`forward`](SabrPricer::forward), which is the default's
/// $Se^{(r-q)\tau}$ under another name.
impl VanillaEuropeanCall for SabrPricer {}
