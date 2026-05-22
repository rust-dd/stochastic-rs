use crate::calibration::sabr::SabrParams;

#[derive(Clone, Copy, Debug)]
pub struct SabrSmileQuotes {
  /// Time to maturity in years
  pub tau: f64,
  /// ATM vol (decimal)
  pub sigma_atm: f64,
  /// Risk-reversal (decimal): sigma(25d call) - sigma(25d put)
  pub sigma_rr: f64,
  /// Butterfly (decimal): average of call/put away-from-atm vol premium over ATM
  pub sigma_bf: f64,
}

#[derive(Clone, Debug)]
pub struct SabrSmileCalibrator {
  /// Spot FX rate S
  pub s: f64,
  /// Domestic rate r_d
  pub r_d: f64,
  /// Foreign rate r_f
  pub r_f: f64,
  /// Cev exponent (0 = normal, 1 = lognormal)
  pub beta: f64,
  /// Quotes for one tenor
  pub quotes: SabrSmileQuotes,
  /// Tau threshold below which the calibrator uses extra basin-hopping
  /// iterations. The Hagan expansion is steeper near zero-tau, so short
  /// tenors need a longer search. Default: `7/365` (~1 trading week).
  pub short_tenor_threshold: f64,
  /// Basin-hopping iterations when `tau < short_tenor_threshold`.
  /// Default: 1000.
  pub short_tenor_iters: usize,
  /// Basin-hopping iterations when `tau >= short_tenor_threshold`.
  /// Default: 100.
  pub long_tenor_iters: usize,
  /// Minimum strike for `[k_rr_c, k_rr_p, k_bf_c, k_bf_p]`. Defaults to
  /// `s * 0.5`. Override via [`Self::with_strike_bounds`] for low-spot
  /// underlyings (penny FX) where `s * 0.5` is too tight, or for non-FX
  /// underlyings (equity / commodity) where the FX-style multiplicative
  /// bounds are wrong.
  pub strike_lo: f64,
  /// Maximum strike for `[k_rr_c, k_rr_p, k_bf_c, k_bf_p]`. Defaults to
  /// `s * 2.0`.
  pub strike_hi: f64,
  /// Tolerance on the final objective for [`SabrSmileResult::success`] to be
  /// `true`. Default: `1e-3`. The previous "is finite" predicate marked
  /// nonsense fits as successful — use this threshold to surface real
  /// non-convergence to callers.
  pub success_tol: f64,
}

impl SabrSmileCalibrator {
  pub fn new(s: f64, r_d: f64, r_f: f64, beta: f64, quotes: SabrSmileQuotes) -> Self {
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

  /// Override the per-strike bounds for `[k_rr_c, k_rr_p, k_bf_c, k_bf_p]`.
  /// The defaults `(s * 0.5, s * 2.0)` work for typical equity / FX with a
  /// well-defined spot. For penny FX, deep-OTM commodities, or any
  /// non-multiplicative regime, supply explicit bounds.
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

#[derive(Clone, Debug)]
pub struct SabrSmileResult {
  /// ATM strike (= forward).
  pub k_atm: f64,
  /// Call strike corresponding to risk-reversal quote.
  pub k_rr_call: f64,
  /// Put strike corresponding to risk-reversal quote.
  pub k_rr_put: f64,
  /// Call strike corresponding to butterfly quote.
  pub k_bf_call: f64,
  /// Put strike corresponding to butterfly quote.
  pub k_bf_put: f64,
  /// Model parameter set (calibrated output).
  pub params: SabrParams,
  /// Final objective-function value.
  pub objective: f64,
  /// Indicates whether optimization converged successfully.
  pub success: bool,
}
