use crate::QuantError;

#[derive(Clone, Copy, Debug)]
pub struct SabrSmileQuotes {
  /// Time to maturity in years.
  pub tau: f64,
  /// ATM vol (decimal)
  pub sigma_atm: f64,
  /// Risk-reversal (decimal): sigma(25d call) - sigma(25d put)
  pub sigma_rr: f64,
  /// Butterfly (decimal): average of call/put away-from-atm vol premium over ATM
  pub sigma_bf: f64,
}

impl SabrSmileQuotes {
  /// `InvalidInput` unless `tau` and `sigma_atm` are finite and positive, both spreads finite and
  /// the strangle's quoting vol `sigma_atm + sigma_bf` positive.
  pub(super) fn validate(&self) -> Result<(), QuantError> {
    let Self {
      tau,
      sigma_atm,
      sigma_rr,
      sigma_bf,
    } = *self;
    let reason = if !(tau.is_finite() && tau > 0.0) {
      format!("tau must satisfy `tau.is_finite() && tau > 0.0`, got tau = {tau:?}")
    } else if !(sigma_atm.is_finite() && sigma_atm > 0.0) {
      format!(
        "sigma_atm must satisfy `sigma_atm.is_finite() && sigma_atm > 0.0`, got sigma_atm = \
         {sigma_atm:?}"
      )
    } else if !sigma_rr.is_finite() {
      format!("sigma_rr must satisfy `sigma_rr.is_finite()`, got sigma_rr = {sigma_rr:?}")
    } else if !sigma_bf.is_finite() {
      format!("sigma_bf must satisfy `sigma_bf.is_finite()`, got sigma_bf = {sigma_bf:?}")
    } else if sigma_atm + sigma_bf <= 0.0 {
      format!(
        "sigma_atm + sigma_bf must satisfy `sigma_atm + sigma_bf > 0.0`, got sigma_atm + sigma_bf = \
         {:?}",
        sigma_atm + sigma_bf
      )
    } else {
      return Ok(());
    };
    Err(QuantError::invalid("smile quotes", reason))
  }
}

/// The strikes a smile fit solves for: ATM (the forward) and the 25-delta risk-reversal and
/// butterfly call/put pairs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmileStrikes {
  pub k_atm: f64,
  pub k_rr_call: f64,
  pub k_rr_put: f64,
  pub k_bf_call: f64,
  pub k_bf_put: f64,
}

#[derive(Clone, Debug)]
pub struct SabrSmileCalibrator {
  /// Spot FX rate.
  pub(super) s: f64,
  /// Domestic rate r_d
  pub(super) r_d: f64,
  /// Foreign rate r_f
  pub(super) r_f: f64,
  /// Cev exponent (0 = normal, 1 = lognormal)
  pub beta: f64,
  /// Quotes for one tenor
  pub quotes: SabrSmileQuotes,
  /// `tau` below which the calibrator runs extra basin-hopping iterations, as Hagan's expansion
  /// steepens near `tau = 0`; `7/365` by default.
  pub short_tenor_threshold: f64,
  /// Basin-hopping iterations when `tau < short_tenor_threshold`.
  /// Default: 1000.
  pub short_tenor_iters: usize,
  /// Basin-hopping iterations when `tau >= short_tenor_threshold`.
  /// Default: 100.
  pub long_tenor_iters: usize,
  /// Lower bound of `[k_rr_c, k_rr_p, k_bf_c, k_bf_p]`, `s · 0.5` by default; see
  /// [`Self::with_strike_bounds`] for penny FX or a non-FX underlying.
  pub strike_lo: f64,
  /// Maximum strike for `[k_rr_c, k_rr_p, k_bf_c, k_bf_p]`. Defaults to
  /// `s * 2.0`.
  pub strike_hi: f64,
  /// `converged` holds when the final objective is below this; default `1e-3`.
  pub success_tol: f64,
}

impl SabrSmileCalibrator {
  /// Panics unless `s` is finite and positive, `r_d` and `r_f` are finite and `beta ∈ [0, 1]`.
  pub fn new(s: f64, r_d: f64, r_f: f64, beta: f64, quotes: SabrSmileQuotes) -> Self {
    assert!(
      s.is_finite() && s > 0.0,
      "s must satisfy `s.is_finite() && s > 0.0`, got s = {s:?}"
    );
    assert!(
      r_d.is_finite(),
      "r_d must satisfy `r_d.is_finite()`, got r_d = {r_d:?}"
    );
    assert!(
      r_f.is_finite(),
      "r_f must satisfy `r_f.is_finite()`, got r_f = {r_f:?}"
    );
    assert!(
      (0.0..=1.0).contains(&beta),
      "beta must satisfy `(0.0..=1.0).contains(&beta)`, got beta = {beta:?}"
    );
    Self {
      s,
      r_d,
      r_f,
      beta,
      quotes,
      short_tenor_threshold: 7.0 / 365.0,
      short_tenor_iters: 1000,
      long_tenor_iters: 100,
      strike_lo: s * 0.5,
      strike_hi: s * 2.0,
      success_tol: 1e-3,
    }
  }

  pub fn s(&self) -> f64 {
    self.s
  }

  pub fn r_d(&self) -> f64 {
    self.r_d
  }

  pub fn r_f(&self) -> f64 {
    self.r_f
  }

  /// Override the short-tenor cutoff (in years).
  pub fn with_short_tenor_threshold(mut self, threshold: f64) -> Self {
    self.short_tenor_threshold = threshold;
    self
  }

  /// Override basin-hopping iterations for short and long tenor regimes.
  pub fn with_basin_hopping_iters(mut self, short: usize, long: usize) -> Self {
    self.short_tenor_iters = short;
    self.long_tenor_iters = long;
    self
  }

  /// Bounds for `[k_rr_c, k_rr_p, k_bf_c, k_bf_p]` in place of `(s · 0.5, s · 2.0)`, for penny FX,
  /// deep out-of-the-money commodities or any non-multiplicative regime.
  pub fn with_strike_bounds(mut self, lo: f64, hi: f64) -> Self {
    assert!(
      lo > 0.0 && hi > lo,
      "strike bounds must satisfy 0 < lo < hi"
    );
    self.strike_lo = lo;
    self.strike_hi = hi;
    self
  }

  /// Override the success tolerance applied to the final objective.
  pub fn with_success_tol(mut self, tol: f64) -> Self {
    assert!(tol >= 0.0, "success_tol must be >= 0");
    self.success_tol = tol;
    self
  }
}
