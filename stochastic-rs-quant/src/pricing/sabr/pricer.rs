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
  /// possibly non-positive on legal parameters (see [`price_call`](Self::price_call)).
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

  /// The price at the query's forward `f`, for a fit that prices each quote at many parameter
  /// sets and computes its forward once; NaN as [`price_call`](Self::price_call) is.
  // One argument per term of the query, the forward computed once.
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn price_at_forward(
    &self,
    f: f64,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    option_type: OptionType,
  ) -> f64 {
    let sigma = hagan_implied_vol(k, f, tau, &self.params);
    if !sigma.is_finite() || sigma <= 0.0 {
      return f64::NAN;
    }
    BsmPricer::new(sigma, BsmCoc::Merton1973).price_option(s, k, r, q, tau, option_type)
  }

  /// Black volatility implied by `c_price` under the carry `b = r - q`, whatever the SABR
  /// parameters; NaN for a price outside the no-arbitrage bounds.
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
    let forward = self.forward(s, r, q, tau);
    let undiscounted_price = c_price * (r * tau).exp();
    crate::vol_surface::implied::black_implied_vol(undiscounted_price, forward, k, tau, option_type)
  }
}

impl ModelPricer for SabrPricer {
  /// NaN unless [`sigma`](SabrPricer::sigma) is finite and > 0, as where Hagan's small-τ bracket
  /// turns negative (`α 0.2, β 1, ρ −0.9, ν 3` at `τ = 10`).
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price_at_forward(
      self.forward(s, r, q, tau),
      s,
      k,
      r,
      q,
      tau,
      OptionType::Call,
    )
  }

  /// The Black-Scholes put at the Hagan vol, not the parity default, which rounds differently;
  /// NaN where [`price_call`](SabrPricer::price_call) is.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price_at_forward(self.forward(s, r, q, tau), s, k, r, q, tau, OptionType::Put)
  }
}

/// European vanilla call: the Black price at Hagan's volatility against
/// [`forward`](SabrPricer::forward), the default's $Se^{(r-q)\tau}$.
impl VanillaEuropeanCall for SabrPricer {}
