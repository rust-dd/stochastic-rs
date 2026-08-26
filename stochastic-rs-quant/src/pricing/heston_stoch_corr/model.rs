//! `HestonStochCorrPricer` model struct and its `ModelPricer` impl.
//! Characteristic-function logic lives in [`super::cf`], Carr-Madan
//! inversion in [`super::pricer`].

use super::pricer::floor_price;
use crate::OptionType;
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
    assert!(
      v0 >= 0.0,
      "HestonStochCorrPricer::new: v0 must be a non-negative variance (got {v0})"
    );
    assert!(
      theta_v >= 0.0,
      "HestonStochCorrPricer::new: theta_v must be a non-negative variance (got {theta_v})"
    );
    assert!(
      sigma_v >= 0.0,
      "HestonStochCorrPricer::new: sigma_v must be a non-negative volatility (got {sigma_v})"
    );
    assert!(
      sigma_r >= 0.0,
      "HestonStochCorrPricer::new: sigma_r must be a non-negative volatility (got {sigma_r})"
    );
    assert!(
      (-1.0..=1.0).contains(&rho0),
      "HestonStochCorrPricer::new: rho0 must be in [-1, 1] (got {rho0})"
    );
    assert!(
      (-1.0..=1.0).contains(&mu_r),
      "HestonStochCorrPricer::new: mu_r must be in [-1, 1] (got {mu_r})"
    );
    assert!(
      (-1.0..=1.0).contains(&rho2),
      "HestonStochCorrPricer::new: rho2 must be in [-1, 1] (got {rho2})"
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

  /// Call and put price at one query point. Both legs are floored at zero:
  /// the Carr-Madan inversion is a numerical quadrature and can return a
  /// small negative value deep out of the money.
  ///
  /// Returns `(NaN, NaN)` when any of `s`, `k`, `r`, `q` or `tau` is
  /// non-finite. The floor tests for `NaN` before it clamps — see
  /// `floor_price` in [`super::pricer`] — because `f64::max` would otherwise
  /// turn an undefined price into a plausible zero.
  pub fn call_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> (f64, f64) {
    let call = self.price_call_carr_madan(s, k, r, q, tau);
    let put = call + k * (-r * tau).exp() - s * (-q * tau).exp();

    (floor_price(call), floor_price(put))
  }

  /// Black volatility implied by `price` at one query point.
  ///
  /// Depends on none of this model's own parameters — it inverts a price
  /// for a volatility rather than pricing at one — but is kept here as the
  /// inverse of [`call_put`](Self::call_put), sharing its `b = r - q` carry
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
    use implied_vol::DefaultSpecialFn;
    use implied_vol::ImpliedBlackVolatility;

    let forward = s * ((r - q) * tau).exp();
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

impl ModelPricer for HestonStochCorrPricer {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put(s, k, r, q, tau).0
  }

  /// Overrides the trait's vanilla-parity default. The arithmetic *is*
  /// parity — this model's carry factor really is $e^{-q\tau}$ — but the
  /// default drops the `max(0)` floor that
  /// [`call_put`](HestonStochCorrPricer::call_put) applies to both legs,
  /// and associates the three terms in a different order from the
  /// pre-query `calculate_call_put`. Routing through `call_put` keeps both.
  /// See `hscm_put_is_parity_and_is_floored_at_zero`.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put(s, k, r, q, tau).1
  }
}

/// European vanilla call on the forward $Se^{(r-q)\tau}$ — the carry is
/// $b=r-q$, the same case
/// [`implied_volatility`](HestonStochCorrPricer::implied_volatility) already
/// inverts against.
impl VanillaEuropeanCall for HestonStochCorrPricer {}
