//! Vanna–Volga smile construction from FX market quotes.
//!
//! Given the canonical FX vol triplet — ATM volatility plus 25-delta risk
//! reversal (RR) and butterfly (BF) — the first-order Vanna–Volga method
//! interpolates a quadratic-in-log-strike smile through three pivot
//! strikes: 25Δ put, ATM, 25Δ call.
//!
//! Reference: Castagna, "FX Options and Smile Risk", Wiley (2010), §3;
//! Reiswich & Wystup, "FX Volatility Smile Construction", Wilmott (2010);
//! Perederiy, "Vanna-Volga Method for Normal Volatilities",
//! arXiv:1810.07457 (2018).
//!
//! Standard market triplet decomposition:
//! - $\sigma_{25C} = \sigma_{ATM} + \sigma_{BF} + \tfrac12\,\sigma_{RR}$
//! - $\sigma_{25P} = \sigma_{ATM} + \sigma_{BF} - \tfrac12\,\sigma_{RR}$

use super::delta::AtmConvention;
use super::delta::FxDeltaConvention;
use super::delta::atm_strike;
use super::delta::strike_from_delta;
use crate::OptionType;

/// FX market quote bundle (ATM + 25Δ risk reversal + 25Δ butterfly).
///
/// All vols are expressed in absolute units (e.g. `0.10` for 10 % vol).
/// `rr` and `bf` follow the standard FX market quoting:
/// `rr = σ(25Δ call) - σ(25Δ put)` and
/// `bf = ½ (σ(25Δ call) + σ(25Δ put)) - σ(ATM)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FxMarketQuotes {
  pub atm: f64,
  pub rr_25: f64,
  pub bf_25: f64,
  pub atm_convention: AtmConvention,
  pub delta_convention: FxDeltaConvention,
}

impl FxMarketQuotes {
  /// 25Δ call vol from the triplet.
  pub fn vol_call_25(&self) -> f64 {
    self.atm + self.bf_25 + 0.5 * self.rr_25
  }

  /// 25Δ put vol from the triplet.
  pub fn vol_put_25(&self) -> f64 {
    self.atm + self.bf_25 - 0.5 * self.rr_25
  }
}

/// Vanna–Volga smile from market quotes pinned at three strikes.
#[derive(Debug, Clone, Copy)]
pub struct VannaVolgaSmile {
  pub k_put: f64,
  pub k_atm: f64,
  pub k_call: f64,
  pub vol_put: f64,
  pub vol_atm: f64,
  pub vol_call: f64,
}

impl VannaVolgaSmile {
  /// Build the smile by solving for the three pivot strikes from market
  /// quotes and the supplied forward / rate environment.
  pub fn build(quotes: FxMarketQuotes, forward: f64, tau: f64, r_f: f64) -> Self {
    let k_atm = atm_strike(forward, quotes.atm, tau, quotes.atm_convention);

    let k_call = strike_from_delta(
      0.25,
      forward,
      quotes.vol_call_25(),
      tau,
      r_f,
      OptionType::Call,
      quotes.delta_convention,
    );
    let k_put = strike_from_delta(
      -0.25,
      forward,
      quotes.vol_put_25(),
      tau,
      r_f,
      OptionType::Put,
      quotes.delta_convention,
    );

    Self {
      k_put,
      k_atm,
      k_call,
      vol_put: quotes.vol_put_25(),
      vol_atm: quotes.atm,
      vol_call: quotes.vol_call_25(),
    }
  }

  /// Implied vol at strike `k` via Lagrange interpolation in log-strike
  /// space (first-order Vanna–Volga).
  ///
  /// Coincides with the market quote at the three pivot strikes; smooth
  /// extrapolation beyond extremes is well-defined but loses any vol
  /// guarantee — production use should clamp or fall back to flat-extrap.
  pub fn vol_at_strike(&self, k: f64) -> f64 {
    debug_assert!(k > 0.0);
    let lk = k.ln();
    let l1 = self.k_put.ln();
    let l2 = self.k_atm.ln();
    let l3 = self.k_call.ln();

    let y1 = ((lk - l2) * (lk - l3)) / ((l1 - l2) * (l1 - l3));
    let y2 = ((lk - l1) * (lk - l3)) / ((l2 - l1) * (l2 - l3));
    let y3 = ((lk - l1) * (lk - l2)) / ((l3 - l1) * (l3 - l2));

    y1 * self.vol_put + y2 * self.vol_atm + y3 * self.vol_call
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn quotes() -> FxMarketQuotes {
    FxMarketQuotes {
      atm: 0.10,
      rr_25: -0.005,
      bf_25: 0.0015,
      atm_convention: AtmConvention::DeltaNeutralStraddle,
      delta_convention: FxDeltaConvention::Forward,
    }
  }

  #[test]
  fn triplet_decomposition_consistent() {
    let q = quotes();
    let half_sum = 0.5 * (q.vol_call_25() + q.vol_put_25());
    assert!((half_sum - (q.atm + q.bf_25)).abs() < 1e-15);
    let diff = q.vol_call_25() - q.vol_put_25();
    assert!((diff - q.rr_25).abs() < 1e-15);
  }

  #[test]
  fn smile_passes_through_pivots() {
    let q = quotes();
    let s = VannaVolgaSmile::build(q, 1.10, 0.5, 0.02);
    assert!((s.vol_at_strike(s.k_put) - q.vol_put_25()).abs() < 1e-12);
    assert!((s.vol_at_strike(s.k_atm) - q.atm).abs() < 1e-12);
    assert!((s.vol_at_strike(s.k_call) - q.vol_call_25()).abs() < 1e-12);
  }

  #[test]
  fn smile_strikes_are_ordered() {
    let q = quotes();
    let s = VannaVolgaSmile::build(q, 1.10, 0.5, 0.02);
    assert!(s.k_put < s.k_atm, "K_put={} K_atm={}", s.k_put, s.k_atm);
    assert!(s.k_atm < s.k_call, "K_atm={} K_call={}", s.k_atm, s.k_call);
  }

  #[test]
  fn smile_smooth_between_pivots() {
    let q = quotes();
    let s = VannaVolgaSmile::build(q, 1.10, 0.5, 0.02);
    // Between K_put and K_atm, vol should land between min and max of pivots.
    let k_mid = (s.k_put * s.k_atm).sqrt();
    let v_mid = s.vol_at_strike(k_mid);
    let lo = s.vol_put.min(s.vol_atm).min(s.vol_call) - 5e-4;
    let hi = s.vol_put.max(s.vol_atm).max(s.vol_call) + 5e-4;
    assert!(v_mid >= lo && v_mid <= hi, "mid vol = {v_mid}");
  }

  #[test]
  fn flat_smile_when_rr_and_bf_zero() {
    let q = FxMarketQuotes {
      atm: 0.12,
      rr_25: 0.0,
      bf_25: 0.0,
      atm_convention: AtmConvention::Forward,
      delta_convention: FxDeltaConvention::Forward,
    };
    let s = VannaVolgaSmile::build(q, 1.10, 0.5, 0.02);
    for &k in &[0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25] {
      assert!((s.vol_at_strike(k) - 0.12).abs() < 1e-10, "k={k}");
    }
  }
}
