//! FX delta conventions and ATM straddle conventions.
//!
//! FX option markets quote volatility against *delta* rather than strike,
//! and four delta conventions coexist: spot vs forward and "raw" vs
//! premium-adjusted. ATM volatility additionally has multiple
//! interpretations (forward, delta-neutral straddle, premium-adjusted DNS).
//!
//! Reference: D. Reiswich & U. Wystup, "FX Volatility Smile Construction",
//! Wilmott Magazine (2010); Castagna, "FX Options and Smile Risk", Wiley (2010);
//! Clark, "Foreign Exchange Option Pricing", Wiley (2011).
//!
//! Notation: forward $F = S e^{(r_d - r_f)\tau}$, base discount
//! $D_f = e^{-r_f \tau}$.

use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::special::norm_cdf;

use crate::OptionType;

/// FX delta convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FxDeltaConvention {
  /// Spot delta with no premium adjustment: $\Delta_{\text{spot}} = e^{-r_f\tau}\,\Phi(d_1)$.
  /// Standard for short-dated options (≤ 1y) outside JPY pairs.
  Spot,
  /// Forward delta: $\Delta_{\text{fwd}} = \Phi(d_1)$.
  /// Standard for long-dated options.
  Forward,
  /// Premium-adjusted spot delta:
  /// $\Delta_{\text{spot,pa}} = e^{-r_f\tau}\,\frac{K}{F}\,\Phi(d_2)$.
  /// Standard for currency pairs where the premium is paid in the base
  /// currency (e.g. EUR/USD with USD-denominated premium).
  SpotPremiumAdjusted,
  /// Premium-adjusted forward delta:
  /// $\Delta_{\text{fwd,pa}} = \frac{K}{F}\,\Phi(d_2)$.
  ForwardPremiumAdjusted,
}

/// At-the-money convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtmConvention {
  /// At-the-money forward: $K = F$.
  Forward,
  /// Delta-neutral straddle (spot/forward delta): $K = F\,e^{\sigma^2\tau/2}$.
  /// Strike at which $\Delta_{\text{call}}(K) = -\Delta_{\text{put}}(K)$.
  DeltaNeutralStraddle,
  /// Premium-adjusted delta-neutral straddle:
  /// $K = F\,e^{-\sigma^2\tau/2}$.
  DeltaNeutralStraddlePremiumAdjusted,
}

/// Compute delta of a vanilla European FX option under the given convention.
///
/// `forward` is $F = S e^{(r_d - r_f)\tau}$, `r_f` is the foreign
/// (base-currency) rate, `tau` is time to maturity in years.
pub fn delta(
  strike: f64,
  forward: f64,
  sigma: f64,
  tau: f64,
  r_f: f64,
  option_type: OptionType,
  convention: FxDeltaConvention,
) -> f64 {
  let sqrt_t = tau.sqrt();
  let d1 = ((forward / strike).ln() + 0.5 * sigma * sigma * tau) / (sigma * sqrt_t);
  let d2 = d1 - sigma * sqrt_t;
  let df = (-r_f * tau).exp();
  let phi_d1 = match option_type {
    OptionType::Call => norm_cdf(d1),
    OptionType::Put => norm_cdf(d1) - 1.0,
  };
  let phi_d2 = match option_type {
    OptionType::Call => norm_cdf(d2),
    OptionType::Put => norm_cdf(d2) - 1.0,
  };
  match convention {
    FxDeltaConvention::Spot => df * phi_d1,
    FxDeltaConvention::Forward => phi_d1,
    FxDeltaConvention::SpotPremiumAdjusted => df * (strike / forward) * phi_d2,
    FxDeltaConvention::ForwardPremiumAdjusted => (strike / forward) * phi_d2,
  }
}

/// Solve for the strike implied by a delta quote.
///
/// `target_delta` is signed — for a 25Δ call pass `0.25`, for a 25Δ put
/// pass `-0.25`. For non-premium-adjusted conventions a closed-form
/// inverse exists via $\Phi^{-1}$; premium-adjusted conventions require
/// a 1-D root solve since the delta depends on $K$ through both $d_1$
/// and the explicit $K/F$ factor.
pub fn strike_from_delta(
  target_delta: f64,
  forward: f64,
  sigma: f64,
  tau: f64,
  r_f: f64,
  option_type: OptionType,
  convention: FxDeltaConvention,
) -> f64 {
  assert!(sigma > 0.0 && tau > 0.0 && forward > 0.0);
  let sqrt_t = tau.sqrt();
  let df = (-r_f * tau).exp();
  let sign = match option_type {
    OptionType::Call => 1.0,
    OptionType::Put => -1.0,
  };
  match convention {
    FxDeltaConvention::Spot => {
      // df·sign·Φ(sign·d1) = target_delta  ⇒  d1 = sign·Φ⁻¹(|target|/df).
      let phi_arg = (target_delta / df / sign).clamp(1e-12, 1.0 - 1e-12);
      let d1 = sign * ndtri(phi_arg);
      forward * (-d1 * sigma * sqrt_t + 0.5 * sigma * sigma * tau).exp()
    }
    FxDeltaConvention::Forward => {
      let phi_arg = (target_delta / sign).clamp(1e-12, 1.0 - 1e-12);
      let d1 = sign * ndtri(phi_arg);
      forward * (-d1 * sigma * sqrt_t + 0.5 * sigma * sigma * tau).exp()
    }
    FxDeltaConvention::SpotPremiumAdjusted | FxDeltaConvention::ForwardPremiumAdjusted => {
      brent_strike(target_delta, forward, sigma, tau, r_f, option_type, convention)
    }
  }
}

/// ATM strike under the given convention.
pub fn atm_strike(forward: f64, sigma: f64, tau: f64, convention: AtmConvention) -> f64 {
  match convention {
    AtmConvention::Forward => forward,
    AtmConvention::DeltaNeutralStraddle => forward * (0.5 * sigma * sigma * tau).exp(),
    AtmConvention::DeltaNeutralStraddlePremiumAdjusted => {
      forward * (-0.5 * sigma * sigma * tau).exp()
    }
  }
}

fn brent_strike(
  target_delta: f64,
  forward: f64,
  sigma: f64,
  tau: f64,
  r_f: f64,
  option_type: OptionType,
  convention: FxDeltaConvention,
) -> f64 {
  // Premium-adjusted call delta is non-monotonic in K — peaks then falls
  // back toward 0. The market convention selects the **larger** strike
  // root (the OTM side), so bracket the side where K is on the OTM tail.
  // Premium-adjusted put delta is monotonic on K < F.
  let (mut lo, mut hi) = match option_type {
    OptionType::Call => (forward, forward * 1e4),
    OptionType::Put => (forward * 1e-4, forward),
  };
  let mut f_lo = delta(lo, forward, sigma, tau, r_f, option_type, convention) - target_delta;
  let mut f_hi = delta(hi, forward, sigma, tau, r_f, option_type, convention) - target_delta;
  if f_lo * f_hi > 0.0 {
    return f64::NAN;
  }
  for _ in 0..200 {
    let mid = 0.5 * (lo + hi);
    let f_mid =
      delta(mid, forward, sigma, tau, r_f, option_type, convention) - target_delta;
    if f_mid.abs() < 1e-12 || (hi - lo) / mid < 1e-12 {
      return mid;
    }
    if f_lo * f_mid < 0.0 {
      hi = mid;
      f_hi = f_mid;
    } else {
      lo = mid;
      f_lo = f_mid;
    }
    let _ = f_hi;
  }
  0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
  use super::*;

  const F: f64 = 1.10;
  const SIGMA: f64 = 0.10;
  const TAU: f64 = 0.5;
  const R_F: f64 = 0.02;

  #[test]
  fn forward_atm_strike_is_forward() {
    assert!((atm_strike(F, SIGMA, TAU, AtmConvention::Forward) - F).abs() < 1e-15);
  }

  #[test]
  fn dns_strike_is_above_forward_for_positive_vol() {
    let k = atm_strike(F, SIGMA, TAU, AtmConvention::DeltaNeutralStraddle);
    assert!(k > F);
    let expected = F * (0.5 * SIGMA * SIGMA * TAU).exp();
    assert!((k - expected).abs() < 1e-12);
  }

  #[test]
  fn dns_pa_strike_is_below_forward() {
    let k = atm_strike(F, SIGMA, TAU, AtmConvention::DeltaNeutralStraddlePremiumAdjusted);
    assert!(k < F);
  }

  #[test]
  fn forward_delta_call_25_round_trip() {
    let k = strike_from_delta(0.25, F, SIGMA, TAU, R_F, OptionType::Call, FxDeltaConvention::Forward);
    let d = delta(k, F, SIGMA, TAU, R_F, OptionType::Call, FxDeltaConvention::Forward);
    assert!((d - 0.25).abs() < 5e-7, "k={k} d={d}");
  }

  #[test]
  fn spot_delta_put_25_round_trip() {
    let k = strike_from_delta(-0.25, F, SIGMA, TAU, R_F, OptionType::Put, FxDeltaConvention::Spot);
    let d = delta(k, F, SIGMA, TAU, R_F, OptionType::Put, FxDeltaConvention::Spot);
    assert!((d + 0.25).abs() < 5e-7, "k={k} d={d}");
  }

  #[test]
  fn spot_call_delta_smaller_than_forward_call_delta() {
    let k = F;
    let d_spot = delta(k, F, SIGMA, TAU, R_F, OptionType::Call, FxDeltaConvention::Spot);
    let d_fwd = delta(k, F, SIGMA, TAU, R_F, OptionType::Call, FxDeltaConvention::Forward);
    assert!(d_spot < d_fwd, "spot Δ ({d_spot}) should be smaller than forward Δ ({d_fwd})");
  }

  #[test]
  fn premium_adjusted_spot_delta_round_trip() {
    let k = strike_from_delta(
      0.25,
      F,
      SIGMA,
      TAU,
      R_F,
      OptionType::Call,
      FxDeltaConvention::SpotPremiumAdjusted,
    );
    let d = delta(
      k,
      F,
      SIGMA,
      TAU,
      R_F,
      OptionType::Call,
      FxDeltaConvention::SpotPremiumAdjusted,
    );
    assert!((d - 0.25).abs() < 5e-7, "round-trip Δ={d}");
  }

  #[test]
  fn dns_strike_is_call_put_delta_neutral() {
    // K = F·exp(σ²τ/2) ⇒ d1 = σ√τ/2, so Φ(d1) - Φ(-d1) = symmetric in
    // forward delta convention.
    let k = atm_strike(F, SIGMA, TAU, AtmConvention::DeltaNeutralStraddle);
    let d_call = delta(k, F, SIGMA, TAU, R_F, OptionType::Call, FxDeltaConvention::Forward);
    let d_put = delta(k, F, SIGMA, TAU, R_F, OptionType::Put, FxDeltaConvention::Forward);
    assert!((d_call + d_put).abs() < 1e-8, "Δ_call + Δ_put = {} (expected 0)", d_call + d_put);
  }
}
