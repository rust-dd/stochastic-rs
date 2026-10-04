use implied_vol::DefaultSpecialFn;
use implied_vol::ImpliedBlackVolatility;

use crate::OptionType;
use crate::pricing::bsm::BSMCoc;
use crate::pricing::bsm::BSMPricer;
use crate::pricing::sabr::hagan::forward_fx;
use crate::pricing::sabr::hagan::fx_delta_from_forward;
use crate::pricing::sabr::hagan::hagan_implied_vol;
use crate::traits::ModelPricer;
use crate::traits::VanillaEuropeanCall;

/// Sabr (Hagan 2002, general β) model parameters.
///
/// The struct holds **model state only** — the four Sabr parameters. Spot,
/// strike, rate, dividend yield and maturity are the pricing *query* and
/// travel as arguments to [`ModelPricer::price_call`], so one instance
/// prices a whole strike/maturity grid. Pricing plugs the Hagan (2002)
/// general-β implied vol into Black-Scholes at Merton (1973) cost of carry
/// (`b = r - q`), which under the FX reading is Garman-Kohlhagen with
/// `(r, q) = (r_d, r_f)`.
///
/// This type absorbed the former `SabrModel`, which held these same four
/// fields and priced the same way: once `SabrPricer` stopped bundling
/// market data, the two were the same struct twice. See the 5b report.
///
/// ```
/// use stochastic_rs_quant::pricing::sabr::SabrPricer;
/// use stochastic_rs_quant::traits::ModelPricer;
///
/// let model = SabrPricer::new(0.2, 1.0, 0.4, -0.3);
/// let atm = model.price_call(100.0, 100.0, 0.05, 0.0, 1.0);
/// let otm = model.price_call(100.0, 130.0, 0.05, 0.0, 1.0);
/// assert!(atm > otm);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SabrPricer {
  /// Model shape/loading parameter.
  pub alpha: f64,
  /// Cev exponent (0 = normal, 1 = lognormal).
  pub beta: f64,
  /// Volatility-of-volatility parameter.
  pub nu: f64,
  /// Correlation parameter.
  pub rho: f64,
}

impl SabrPricer {
  /// Validating constructor.
  ///
  /// Two of these four already had a guard and two did not, and the split
  /// was not deliberate: [`hagan_implied_vol`] rejects `alpha` and `rho`,
  /// but one layer down and one call later, while `beta` and `nu` reached
  /// the expansion unchecked and produced finite prices. At `beta = 5` the
  /// at-the-money call returns the spot itself — the no-arbitrage ceiling,
  /// which reads as a legitimate deep-in-the-money value.
  ///
  /// The messages here deliberately avoid
  /// [`hagan_implied_vol`]'s wording, so neither guard's message is a
  /// substring of the other's and a `should_panic` anchored on one cannot
  /// be satisfied by the other firing. The accessor keeps its own check:
  /// the fields are `pub`, so this is a front door and not a wall.
  ///
  /// # Panics
  /// - if `alpha` is not strictly positive, or `NaN` — a volatility level
  /// - if `beta` is outside `[0, 1]`, or `NaN` — the Cev exponent, `0` for
  ///   the normal and `1` for the lognormal model
  /// - if `nu` is negative or `NaN` — a vol-of-vol
  /// - if `rho` is outside the *open* interval `(-1, 1)`, or `NaN`. The
  ///   bound matches [`hagan_implied_vol`]'s exactly: its expansion carries
  ///   a `1 - rho` denominator, so admitting `±1` here would only defer the
  ///   same panic.
  ///
  /// [`SabrCalibrator`](crate::calibration::sabr::SabrCalibrator)'s
  /// projection box — `alpha, nu >= 1e-6`, `beta` in `[0, 1]`, `rho`
  /// clamped to `±0.9999` — lies strictly inside all four, so no legal
  /// calibration iterate can trip this.
  ///
  /// [`hagan_implied_vol`]: crate::pricing::sabr::hagan_implied_vol
  pub fn new(alpha: f64, beta: f64, nu: f64, rho: f64) -> Self {
    assert!(
      alpha > 0.0,
      "SabrPricer::new: alpha must be a strictly positive volatility level (got {alpha})"
    );
    assert!(
      (0.0..=1.0).contains(&beta),
      "SabrPricer::new: beta must be in [0, 1] (got {beta})"
    );
    assert!(
      nu >= 0.0,
      "SabrPricer::new: nu must be a non-negative vol-of-vol (got {nu})"
    );
    assert!(
      rho > -1.0 && rho < 1.0,
      "SabrPricer::new: rho must be in the open interval (-1, 1) (got {rho})"
    );
    Self {
      alpha,
      beta,
      nu,
      rho,
    }
  }

  /// Forward at one query point, `s·e^{(r-q)τ}`. Under the FX reading
  /// `(r, q)` are the domestic and foreign rates.
  pub fn forward(&self, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    forward_fx(s, tau, r, q)
  }

  /// Hagan (2002) general-β implied vol at `k` against [`forward`](Self::forward); NaN for a non-positive `k` or forward or a negative or
  /// non-finite `tau`, possibly non-positive on legal parameters (see [`call_put`](Self::call_put)); panics on an invalid `alpha` / `rho`.
  pub fn sigma(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    hagan_implied_vol(
      k,
      self.forward(s, r, q, tau),
      tau,
      self.alpha,
      self.beta,
      self.nu,
      self.rho,
    )
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

  /// Call and put price at one query point; both legs are NaN when [`sigma`](Self::sigma) is not finite and positive — a query
  /// outside the domain, or a legal parameter set where Hagan's small-τ bracket turns negative (`(0.2, 1, 3, -0.9)` at τ = 10).
  pub fn call_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> (f64, f64) {
    let sigma = self.sigma(s, k, r, q, tau);
    if !sigma.is_finite() || sigma <= 0.0 {
      return (f64::NAN, f64::NAN);
    }
    BSMPricer::new(sigma, BSMCoc::Merton1973).call_put(s, k, r, q, tau)
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
  /// NaN for a query outside the domain or a degenerate Hagan volatility, see [`call_put`](SabrPricer::call_put); panics only on
  /// an invalid `alpha` / `rho` field.
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
