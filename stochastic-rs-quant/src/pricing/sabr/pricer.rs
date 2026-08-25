use implied_vol::DefaultSpecialFn;
use implied_vol::ImpliedBlackVolatility;

use crate::OptionType;
use crate::pricing::bsm::BSMCoc;
use crate::pricing::bsm::BSMPricer;
use crate::pricing::sabr::hagan::forward_fx;
use crate::pricing::sabr::hagan::fx_delta_from_forward;
use crate::pricing::sabr::hagan::hagan_implied_vol;
use crate::traits::ModelPricer;

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
  pub const fn new(alpha: f64, beta: f64, nu: f64, rho: f64) -> Self {
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

  /// Implied volatility from the Hagan (2002) general-β expansion,
  /// evaluated at `k` against [`forward`](Self::forward).
  ///
  /// # Panics
  /// Panics if `k` or the forward is not strictly positive, if
  /// `self.alpha` is not strictly positive, or if `self.rho` does not lie
  /// strictly inside $(-1, 1)$ — see
  /// [`hagan_implied_vol`](crate::pricing::sabr::hagan_implied_vol)'s own
  /// `# Panics` section, which this inherits unchanged. A non-positive spot
  /// or strike is invalid input for this equity/FX model, not a market
  /// state to accommodate, so this panics rather than degrading silently;
  /// [`SabrCalibrator`](crate::calibration::sabr::SabrCalibrator) validates
  /// `s`/`k` before ever constructing a `SabrPricer` from calibrated data,
  /// so this only fires when one is built directly from bad input.
  ///
  /// Validating the arguments does **not** make the result usable as a
  /// volatility: the expansion can still evaluate to a non-positive number
  /// on a legal parameter combination, which is why
  /// [`call_put`](Self::call_put) screens it rather than pricing off it.
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

  /// Call and put price at one query point.
  ///
  /// Returns [`f64::NAN`] for **both** legs when [`sigma`](Self::sigma)
  /// comes out non-finite or not strictly positive, rather than letting a
  /// degenerate volatility propagate into `d1`/`d2`. That is case 2 of the
  /// crate's [failure
  /// convention](crate::traits::ModelPricer#how-pricing-fails) — not
  /// computable here — and not case 1, because every individual argument is
  /// already legal by the time this branch is reachable: `sigma` panics
  /// first on a non-positive `k` or forward, a non-positive `alpha`, or a
  /// `rho` outside $(-1, 1)$.
  ///
  /// What is left is a *parameter combination*. Hagan (2002) is a small-τ
  /// asymptotic expansion whose bracket $1 + (a + b + c)\tau$ turns negative
  /// once the correction term outgrows it, and the ν² coefficient
  /// $c = (2 - 3\rho^2)\nu^2 / 24$ is itself negative for
  /// $|\rho| > \sqrt{2/3}$: at $(\alpha, \beta, \nu, \rho) = (0.2, 1, 3,
  /// -0.9)$ and $\tau = 10$ the expansion returns $\sigma = -0.3925$. Every
  /// one of those four values lies inside
  /// [`SabrCalibrator`](crate::calibration::sabr::SabrCalibrator)'s own
  /// projection box, so this is a calibration-output shape rather than a
  /// user-input one — panicking would abort a whole calibration over a
  /// single bad probe point, where `NaN` marks that residual invalid and
  /// leaves the rest of the grid alone.
  ///
  /// This floored both legs to `0.0` before, the contract the former
  /// `SabrModel` documented. A zero call *and* a zero put is not a price any
  /// instrument has, and unlike `NaN` a zero does not propagate, so a
  /// residual computed against it looked merely bad rather than invalid.
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
  /// # Panics
  /// Panics if `k`, or the forward derived from `s`, is not strictly
  /// positive — see [`sigma`](SabrPricer::sigma)'s `# Panics` section.
  /// [`ModelPricer`] has no fallible return channel, so a non-positive spot
  /// or strike in a strike/maturity grid aborts the whole grid rather than
  /// degrading that one point to `0.0` — deliberately: a non-positive spot
  /// or strike is invalid input for this equity/FX model, not a value worth
  /// pricing as if it were merely deep out-of-the-money.
  ///
  /// Returns [`f64::NAN`] on a degenerate Hagan volatility — see
  /// [`call_put`](SabrPricer::call_put).
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
