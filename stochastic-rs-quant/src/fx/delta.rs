//! FX delta conventions (spot or forward, raw or premium-adjusted) and ATM conventions, with
//! `F = Se^{(r_d − r_f)τ}` and `D_f = e^{−r_f τ}`.
//!
//! Reference: Reiswich & Wystup (2010), "FX Volatility Smile Construction", Wilmott Magazine
//! Reference: Castagna (2010), "FX Options and Smile Risk", Wiley
//! Reference: Clark (2011), "Foreign Exchange Option Pricing", Wiley

use std::f64::consts::FRAC_2_PI;

use stochastic_rs_distributions::brent;
use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_distributions::special::norm_pdf;

use crate::OptionType;
use crate::traits::RealExt;

/// Which delta an FX quote means: spot or forward, with or without the premium adjustment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FxDeltaConvention {
  /// Spot delta with no premium adjustment: $\Delta_{\text{spot}} = e^{-r_f\tau}\,\Phi(d_1)$.
  /// Standard for short-dated options (≤ 1y) outside JPY pairs.
  Spot,
  /// Forward delta: $\Delta_{\text{fwd}} = \Phi(d_1)$.
  /// Standard for long-dated options.
  Forward,
  /// Premium-adjusted spot delta $e^{-r_f\tau}\,\frac{K}{F}\,\Phi(d_2)$, for pairs whose premium
  /// is paid in the base currency.
  SpotPremiumAdjusted,
  /// Premium-adjusted forward delta:
  /// $\Delta_{\text{fwd,pa}} = \frac{K}{F}\,\Phi(d_2)$.
  ForwardPremiumAdjusted,
}

/// At-the-money convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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

/// Delta of a vanilla European FX option under `convention`; `forward` is
/// $F = S e^{(r_d - r_f)\tau}$ and `r_f` the foreign (base-currency) rate.
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
  // A put's −Φ(−d) keeps its digits in the tail, where Φ(d) − 1 cancels.
  let phi_d1 = match option_type {
    OptionType::Call => norm_cdf(d1),
    OptionType::Put => -norm_cdf(-d1),
  };
  let phi_d2 = match option_type {
    OptionType::Call => norm_cdf(d2),
    OptionType::Put => -norm_cdf(-d2),
  };
  match convention {
    FxDeltaConvention::Spot => df * phi_d1,
    FxDeltaConvention::Forward => phi_d1,
    FxDeltaConvention::SpotPremiumAdjusted => df * (strike / forward) * phi_d2,
    FxDeltaConvention::ForwardPremiumAdjusted => (strike / forward) * phi_d2,
  }
}

/// The strike a signed delta implies (`-0.25`: 25Δ put); for a premium-adjusted call, the one past
/// the delta's peak. NaN for a delta no strike attains; panics unless sigma, tau, forward > 0.
pub fn strike_from_delta(
  target_delta: f64,
  forward: f64,
  sigma: f64,
  tau: f64,
  r_f: f64,
  option_type: OptionType,
  convention: FxDeltaConvention,
) -> f64 {
  assert!(
    sigma > 0.0 && tau > 0.0 && forward > 0.0,
    "sigma must satisfy `sigma > 0.0 && tau > 0.0 && forward > 0.0`, got sigma = {sigma:?}, tau = {tau:?}, forward = {forward:?}"
  );
  let df = match convention {
    FxDeltaConvention::Spot | FxDeltaConvention::SpotPremiumAdjusted => (-r_f * tau).exp(),
    FxDeltaConvention::Forward | FxDeltaConvention::ForwardPremiumAdjusted => 1.0,
  };
  let unadjusted = unadjusted_strike(target_delta / df, forward, sigma, tau, option_type);
  match convention {
    FxDeltaConvention::Spot | FxDeltaConvention::Forward => unadjusted,
    FxDeltaConvention::SpotPremiumAdjusted | FxDeltaConvention::ForwardPremiumAdjusted => {
      // A call quote means the root past the delta's peak, a put's the root of a monotonic delta,
      // both at or below the unadjusted strike (Reiswich & Wystup 2010, CPQF WP 20, §1.5).
      let gap = |k: f64| delta(k, forward, sigma, tau, r_f, option_type, convention) - target_delta;
      let (lo, hi) = match option_type {
        OptionType::Call => (peak_strike(forward, sigma * tau.sqrt()), unadjusted),
        OptionType::Put => put_bracket(gap, forward, -target_delta / df, unadjusted),
      };
      brent::zero(lo, hi, f64::MIN_POSITIVE, gap).unwrap_or(f64::NAN)
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

/// The strike whose forward delta is `forward_delta`; NaN outside the open range the delta spans.
fn unadjusted_strike(
  forward_delta: f64,
  forward: f64,
  sigma: f64,
  tau: f64,
  option_type: OptionType,
) -> f64 {
  let sign = match option_type {
    OptionType::Call => 1.0,
    OptionType::Put => -1.0,
  };
  // sign·Φ(sign·d1) = forward_delta  ⇒  d1 = sign·Φ⁻¹(forward_delta / sign).
  let p = forward_delta / sign;
  if !(p > 0.0 && p < 1.0) {
    return f64::NAN;
  }
  let d1 = sign * ndtri(p);
  forward * (-d1 * sigma * tau.sqrt() + 0.5 * sigma * sigma * tau).exp()
}

/// `[1e-4 F, F]` widened to the put root at |Δ| = df·size: Φ(−d₂) ≤ 1 puts it above `size·F`,
/// and past `F`, where Φ(−d₂) > ½, below `2·size·F`; NaN ends unless `size` is finite and positive.
fn put_bracket(gap: impl Fn(f64) -> f64, forward: f64, size: f64, unadjusted: f64) -> (f64, f64) {
  if !(size > 0.0 && size.is_finite()) {
    return (f64::NAN, f64::NAN);
  }
  let (lo, hi) = (forward * 1e-4, forward);
  let lo = if gap(lo) < 0.0 { size * forward } else { lo };
  let hi = if gap(hi) > 0.0 {
    if unadjusted.is_finite() {
      unadjusted
    } else {
      (2.0 * size * forward).max_or_nan(forward)
    }
  } else {
    hi
  };
  (lo, hi)
}

/// Where the premium-adjusted call delta peaks: `d₂` solves `s·Φ(d₂) = φ(d₂)`, `s = σ√τ`.
fn peak_strike(forward: f64, s: f64) -> f64 {
  // Mills' ratio puts the root above −s; it lies below the d where 2φ(d) = s, as Φ(d) > ½ there.
  let hi = (2.0 * (FRAC_2_PI.sqrt() / s).ln()).max_or_nan(0.0).sqrt();
  let d2 = brent::zero(-s, hi, f64::MIN_POSITIVE, |d| s * norm_cdf(d) - norm_pdf(d));
  d2.map_or(f64::NAN, |d2| forward * (-s * d2 - 0.5 * s * s).exp())
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
    let k = atm_strike(
      F,
      SIGMA,
      TAU,
      AtmConvention::DeltaNeutralStraddlePremiumAdjusted,
    );
    assert!(k < F);
  }

  #[test]
  fn forward_delta_call_25_round_trip() {
    let k = strike_from_delta(
      0.25,
      F,
      SIGMA,
      TAU,
      R_F,
      OptionType::Call,
      FxDeltaConvention::Forward,
    );
    let d = delta(
      k,
      F,
      SIGMA,
      TAU,
      R_F,
      OptionType::Call,
      FxDeltaConvention::Forward,
    );
    assert!((d - 0.25).abs() < 5e-7, "k={k} d={d}");
  }

  #[test]
  fn spot_delta_put_25_round_trip() {
    let k = strike_from_delta(
      -0.25,
      F,
      SIGMA,
      TAU,
      R_F,
      OptionType::Put,
      FxDeltaConvention::Spot,
    );
    let d = delta(
      k,
      F,
      SIGMA,
      TAU,
      R_F,
      OptionType::Put,
      FxDeltaConvention::Spot,
    );
    assert!((d + 0.25).abs() < 5e-7, "k={k} d={d}");
  }

  #[test]
  fn spot_call_delta_smaller_than_forward_call_delta() {
    let k = F;
    let d_spot = delta(
      k,
      F,
      SIGMA,
      TAU,
      R_F,
      OptionType::Call,
      FxDeltaConvention::Spot,
    );
    let d_fwd = delta(
      k,
      F,
      SIGMA,
      TAU,
      R_F,
      OptionType::Call,
      FxDeltaConvention::Forward,
    );
    assert!(
      d_spot < d_fwd,
      "spot Δ ({d_spot}) should be smaller than forward Δ ({d_fwd})"
    );
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
  fn a_premium_adjusted_delta_no_strike_reaches_is_nan() {
    // The call delta peaks at 0.838660, at 0.9609 F; a put delta spans (−∞, 0).
    let cases = [
      (OptionType::Call, f64::NAN),
      (OptionType::Call, 0.9),
      (OptionType::Put, 0.0),
      (OptionType::Put, 0.25),
    ];
    for (option_type, target) in cases {
      let convention = FxDeltaConvention::SpotPremiumAdjusted;
      let k = strike_from_delta(target, F, SIGMA, TAU, R_F, option_type, convention);
      assert!(k.is_nan(), "{option_type:?} {target} gave strike {k}");
    }
  }

  #[test]
  fn a_put_delta_in_its_tail_keeps_its_digits() {
    // mpmath's 5e-12 deltas, which Φ(d) − 1 would cancel; d ≈ 6.8 rounds them by 2εd² ≈ 1e-14.
    let cases = [
      (FxDeltaConvention::Spot, -5.331_976_556_076_078e-12),
      (FxDeltaConvention::Forward, -5.385_563_811_353_51e-12),
      (
        FxDeltaConvention::SpotPremiumAdjusted,
        -5.385_807_032_679_407_4e-12,
      ),
      (
        FxDeltaConvention::ForwardPremiumAdjusted,
        -5.439_935_293_240_923e-12,
      ),
    ];
    for (convention, exact) in cases {
      let d = delta(0.682, F, SIGMA, TAU, R_F, OptionType::Put, convention);
      assert!(
        (d - exact).abs() <= 3e-14 * exact.abs(),
        "{convention:?}: {d} vs {exact}"
      );
    }
  }

  #[test]
  fn a_put_strike_for_a_delta_of_1e_12_lands_on_mpmaths_root() {
    // Off mpmath's root by Brent's stop width, 4 macheps K, plus 1e-14 / |KΔ′/Δ| (101) at most;
    // a cancelling Φ(d₂) − 1 would miss it by 4e-7.
    let convention = FxDeltaConvention::SpotPremiumAdjusted;
    let k = strike_from_delta(-1e-12, F, SIGMA, TAU, R_F, OptionType::Put, convention);
    let exact = 0.670_588_058_877_767;
    assert!((k - exact).abs() <= 6e-16 * exact, "{k}");
  }

  #[test]
  fn a_premium_adjusted_put_past_the_old_bracket_lands_on_mpmaths_root() {
    // Above F, past |Δ| = e^{−r_f τ} where no unadjusted strike exists, and below 1e-4 F.
    let (spot, fwd) = (
      FxDeltaConvention::SpotPremiumAdjusted,
      FxDeltaConvention::ForwardPremiumAdjusted,
    );
    let cases = [
      (spot, -0.5, 0.05, 2.0, 0.05, 1.106_878_827_973_364_8),
      (fwd, -0.6, SIGMA, TAU, R_F, 1.115_399_232_640_695_6),
      (spot, -0.95, 0.05, 2.0, 0.05, 1.226_156_765_759_724_5),
      (fwd, -1.2, SIGMA, TAU, R_F, 1.325_057_580_390_090_1),
      (fwd, -1e-5, 2.0, 4.0, R_F, 3.846_795_694_774_183e-5),
    ];
    for (convention, target, sigma, tau, r_f, exact) in cases {
      let k = strike_from_delta(target, F, sigma, tau, r_f, OptionType::Put, convention);
      assert!(
        (k - exact).abs() <= 2e-15 * exact,
        "{convention:?} {target}: {k} vs {exact}"
      );
    }
  }

  #[test]
  fn premium_adjusted_strikes_land_on_mpmaths_roots() {
    use crate::OptionType::Call;
    use crate::OptionType::Put;

    // The flat 5Δ call: a residual-stopping solver misses it by 9e-12, a libm ulp moves it 1.8e-15;
    // the 60Δ call's root past the delta's peak, at 0.9609 F, lies below F.
    let cases = [
      (0.05, 0.8, 2.0, Call, 10.228_143_116_720_913, 4e-15),
      (-0.25, 0.8, 2.0, Put, 0.575_571_943_028_457, 2e-15),
      (0.6, SIGMA, TAU, Call, 1.073_622_697_637_674_2, 2e-15),
    ];
    for (target, sigma, tau, option_type, exact, tolerance) in cases {
      let convention = FxDeltaConvention::SpotPremiumAdjusted;
      let k = strike_from_delta(target, F, sigma, tau, R_F, option_type, convention);
      assert!((k - exact).abs() <= tolerance * exact, "{k} vs {exact}");
    }
  }

  #[test]
  fn a_premium_adjusted_call_takes_the_otm_root_past_a_peak_above_the_forward() {
    let strike = |target| {
      let convention = FxDeltaConvention::ForwardPremiumAdjusted;
      strike_from_delta(target, F, 1.0, 2.0, R_F, OptionType::Call, convention)
    };
    // σ√τ = √2: 0.239750 at F, a peak of 0.241938 at 1.2414 F. At mpmath's root of 0.24,
    // |K Δ′/Δ| = 0.08 turns the delta's few-ulp rounding into up to 1e-14 of the strike.
    let (k, exact) = (strike(0.24), 1.670_988_634_826_725_8);
    assert!((k - exact).abs() <= 1e-14 * exact, "{k}");
    assert!(strike(0.242).is_nan());
  }

  #[test]
  fn a_spot_or_forward_delta_outside_its_range_is_nan() {
    // Forward deltas span (0, 1) and (−1, 0); spot deltas e^{−r_f τ} times those.
    let cases = [
      (FxDeltaConvention::Forward, OptionType::Call, 1.0),
      (FxDeltaConvention::Forward, OptionType::Call, 0.0),
      (FxDeltaConvention::Forward, OptionType::Call, -0.25),
      (FxDeltaConvention::Forward, OptionType::Put, -1.2),
      (FxDeltaConvention::Spot, OptionType::Call, 0.995),
      (FxDeltaConvention::Spot, OptionType::Put, -0.995),
    ];
    for (convention, option_type, target) in cases {
      let k = strike_from_delta(target, F, SIGMA, TAU, R_F, option_type, convention);
      assert!(
        k.is_nan(),
        "{convention:?} {option_type:?} {target} gave strike {k}"
      );
    }
  }

  #[test]
  fn dns_strike_is_call_put_delta_neutral() {
    // K = F·exp(σ²τ/2) ⇒ d1 = σ√τ/2, so Φ(d1) - Φ(-d1) = symmetric in
    // forward delta convention.
    let k = atm_strike(F, SIGMA, TAU, AtmConvention::DeltaNeutralStraddle);
    let d_call = delta(
      k,
      F,
      SIGMA,
      TAU,
      R_F,
      OptionType::Call,
      FxDeltaConvention::Forward,
    );
    let d_put = delta(
      k,
      F,
      SIGMA,
      TAU,
      R_F,
      OptionType::Put,
      FxDeltaConvention::Forward,
    );
    assert!(
      (d_call + d_put).abs() < 1e-8,
      "Δ_call + Δ_put = {} (expected 0)",
      d_call + d_put
    );
  }
}
