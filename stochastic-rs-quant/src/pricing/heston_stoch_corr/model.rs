//! `HestonStochCorrPricer` and its `ModelPricer`; the characteristic function is in [`super::cf`],
//! the Carr–Madan inversion in [`super::pricer`].

use super::pricer::floor_price;
use crate::CallPut;
use crate::OptionType;
use crate::pricing::digital::expired;
use crate::pricing::digital::vanilla_payoff;
use crate::traits::ModelPricer;
use crate::traits::VanillaEuropeanCall;

/// Heston with stochastic correlation (Teng, Ehrhardt & Günther 2016), by Carr–Madan's damped
/// Fourier inversion; the nine model parameters stay, the query travels as arguments.
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
  /// Panics unless `v0`, `theta_v`, `sigma_v`, `sigma_r` are finite and `≥ 0`, the `kappa`s finite,
  /// `κ_v θ_v ≥ 0` and each correlation in `[−1, 1]`; `sigma_v = 0` is the deterministic variance.
  // One argument per parameter of the model.
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
      v0.is_finite() && v0 >= 0.0,
      "v0 must satisfy `v0.is_finite() && v0 >= 0.0`, got v0 = {v0:?}"
    );
    assert!(
      kappa_v.is_finite(),
      "kappa_v must satisfy `kappa_v.is_finite()`, got kappa_v = {kappa_v:?}"
    );
    assert!(
      theta_v.is_finite() && theta_v >= 0.0,
      "theta_v must satisfy `theta_v.is_finite() && theta_v >= 0.0`, got theta_v = {theta_v:?}"
    );
    assert!(
      kappa_v * theta_v >= 0.0,
      "kappa_v must satisfy `kappa_v * theta_v >= 0.0`, got kappa_v = {kappa_v:?}"
    );
    assert!(
      sigma_v.is_finite() && sigma_v >= 0.0,
      "sigma_v must satisfy `sigma_v.is_finite() && sigma_v >= 0.0`, got sigma_v = {sigma_v:?}"
    );
    assert!(
      (-1.0..=1.0).contains(&rho0),
      "rho0 must satisfy `(-1.0..=1.0).contains(&rho0)`, got rho0 = {rho0:?}"
    );
    assert!(
      kappa_r.is_finite(),
      "kappa_r must satisfy `kappa_r.is_finite()`, got kappa_r = {kappa_r:?}"
    );
    assert!(
      (-1.0..=1.0).contains(&mu_r),
      "mu_r must satisfy `(-1.0..=1.0).contains(&mu_r)`, got mu_r = {mu_r:?}"
    );
    assert!(
      sigma_r.is_finite() && sigma_r >= 0.0,
      "sigma_r must satisfy `sigma_r.is_finite() && sigma_r >= 0.0`, got sigma_r = {sigma_r:?}"
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

/// A European vanilla call on `Se^{(r−q)τ}`, the forward
/// [`implied_volatility`](HestonStochCorrPricer::implied_volatility) inverts against.
impl VanillaEuropeanCall for HestonStochCorrPricer {}
