//! `HestonStochCorrPricer` model struct and its `ModelPricer` impl.
//! Characteristic-function logic lives in [`super::cf`], Carr-Madan
//! inversion in [`super::pricer`].

use super::pricer::floor_price;
use crate::CallPut;
use crate::OptionType;
use crate::pricing::digital::expired;
use crate::pricing::digital::vanilla_payoff;
use crate::traits::ModelPricer;
use crate::traits::VanillaEuropeanCall;

/// Heston model with stochastic correlation (Teng, Ehrhardt & Günther,
/// 2016), priced by Carr-Madan dampened Fourier inversion.
///
/// The struct holds **model state only** — the four variance-process
/// parameters, the four correlation-process parameters and the
/// variance/correlation correlation. Spot, strike, rate, dividend yield and
/// maturity are the pricing *query* and travel as arguments to
/// [`ModelPricer::price_call`], so one instance prices a whole
/// strike/maturity grid.
///
/// This type absorbed the former `HscmModel`, which held these same nine
/// fields in the same order and priced through this very struct: once
/// `HestonStochCorrPricer` stopped bundling market data, the two were the
/// same struct twice. See the 5b report.
///
/// ```
/// use stochastic_rs_quant::pricing::heston_stoch_corr::HestonStochCorrPricer;
/// use stochastic_rs_quant::traits::ModelPricer;
///
/// let model = HestonStochCorrPricer::new(
///     0.04, 2.0, 0.04, 0.3, -0.7, 5.0, -0.5, 0.2, 0.3);
/// let atm = model.price_call(100.0, 100.0, 0.05, 0.0, 0.5);
/// let otm = model.price_call(100.0, 120.0, 0.05, 0.0, 0.5);
/// assert!(atm > otm);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct HestonStochCorrPricer {
  // Variance process  dv = κ_v(θ_v − v)dt + σ_v√v dW^v
  /// Initial variance.
  pub v0: f64,
  /// Mean-reversion speed of variance.
  pub kappa_v: f64,
  /// Long-run variance.
  pub theta_v: f64,
  /// Vol-of-vol.
  pub sigma_v: f64,

  // Correlation process  dρ = κ_ρ(μ_ρ − ρ)dt + σ_ρ dW^ρ
  /// Initial correlation.
  pub rho0: f64,
  /// Mean-reversion speed of correlation.
  pub kappa_r: f64,
  /// Long-run correlation level.
  pub mu_r: f64,
  /// Volatility of correlation.
  pub sigma_r: f64,
  /// Correlation between dW^v and dW^ρ.
  pub rho2: f64,
}

impl HestonStochCorrPricer {
  /// Validating constructor.
  ///
  /// Nothing here announced itself before: every invalid value below
  /// produced a finite, plausible Carr-Madan price. A `rho2` of `-1.5` came
  /// within `0.03` of the correct answer, which is the whole problem — it
  /// is indistinguishable from a small modelling difference.
  ///
  /// # Panics
  /// - if `v0` or `theta_v` is negative or `NaN` — a variance cannot be either
  /// - if `sigma_v` or `sigma_r` is negative or `NaN` — a volatility cannot be
  /// - if `rho0`, `mu_r` or `rho2` is outside `[-1, 1]` or `NaN`. All three
  ///   are correlations: the initial level, the level it mean-reverts to
  ///   and the correlation between the two driving Brownians. Leaving any
  ///   one out would put an out-of-range number into the same expansion the
  ///   other two are protected from.
  ///
  /// `sigma_v == 0` is **accepted**, unlike on
  /// [`HestonPricer::new`](crate::pricing::HestonPricer). The reason is the
  /// characteristic function: Heston's closed form divides by $\sigma^2$,
  /// while this model integrates a Riccati system by Rk4 in which
  /// `sigma_v` only ever multiplies, so a zero vol-of-vol is the
  /// deterministic-variance limit rather than a division by zero.
  ///
  /// `kappa_v` and `kappa_r` are unconstrained, matching
  /// [`HestonPricer`](crate::pricing::HestonPricer)'s treatment of
  /// `kappa`. The calibrator's
  /// `BOUNDS` box lies strictly inside every check above, so no legal
  /// calibration iterate can trip this.
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    v0: f64,
    kappa_v: f64,
    theta_v: f64,
    sigma_v: f64,
    rho0: f64,
    kappa_r: f64,
    mu_r: f64,
    sigma_r: f64,
    rho2: f64,
  ) -> Self {
    assert!(v0 >= 0.0, "v0 must satisfy `v0 >= 0.0`, got v0 = {v0:?}");
    assert!(
      theta_v >= 0.0,
      "theta_v must satisfy `theta_v >= 0.0`, got theta_v = {theta_v:?}"
    );
    assert!(
      sigma_v >= 0.0,
      "sigma_v must satisfy `sigma_v >= 0.0`, got sigma_v = {sigma_v:?}"
    );
    assert!(
      sigma_r >= 0.0,
      "sigma_r must satisfy `sigma_r >= 0.0`, got sigma_r = {sigma_r:?}"
    );
    assert!(
      (-1.0..=1.0).contains(&rho0),
      "rho0 must satisfy `(-1.0..=1.0).contains(&rho0)`, got rho0 = {rho0:?}"
    );
    assert!(
      (-1.0..=1.0).contains(&mu_r),
      "mu_r must satisfy `(-1.0..=1.0).contains(&mu_r)`, got mu_r = {mu_r:?}"
    );
    assert!(
      (-1.0..=1.0).contains(&rho2),
      "rho2 must satisfy `(-1.0..=1.0).contains(&rho2)`, got rho2 = {rho2:?}"
    );
    Self {
      v0,
      kappa_v,
      theta_v,
      sigma_v,
      rho0,
      kappa_r,
      mu_r,
      sigma_r,
      rho2,
    }
  }

  /// Call and put from one Carr-Madan inversion, each floored at zero; the payoffs at `tau = 0`;
  /// NaN past expiry and for a non-finite `s`, `k`, `r`, `q` or `tau`.
  pub fn price_call_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> CallPut {
    let payoff = |leg| expired(tau, &[r, q], || vanilla_payoff(leg, s, k));
    if let (Some(call), Some(put)) = (payoff(OptionType::Call), payoff(OptionType::Put)) {
      return CallPut { call, put };
    }
    let call = self.price_call_carr_madan(s, k, r, q, tau);
    let put = call + k * (-r * tau).exp() - s * (-q * tau).exp();

    CallPut {
      call: floor_price(call),
      put: floor_price(put),
    }
  }

  /// Black volatility implied by `c_price` under the carry `b = r - q`, whatever the model's
  /// parameters; NaN for a price outside the no-arbitrage bounds.
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
    crate::vol_surface::implied::black_implied_vol(undiscounted_price, forward, k, tau, option_type)
  }
}

impl ModelPricer for HestonStochCorrPricer {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price_call_put(s, k, r, q, tau).call
  }

  /// The pair's put, floored at zero, which the parity default is not.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price_call_put(s, k, r, q, tau).put
  }
}

/// European vanilla call on the forward $Se^{(r-q)\tau}$ — the carry is
/// $b=r-q$, the same case
/// [`implied_volatility`](HestonStochCorrPricer::implied_volatility) already
/// inverts against.
impl VanillaEuropeanCall for HestonStochCorrPricer {}
