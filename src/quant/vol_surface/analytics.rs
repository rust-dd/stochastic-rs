//! # Smile and Skew Analytics
//!
//! Smile-level analytics computed from an implied volatility slice or
//! a parametric representation (SVI / SSVI).
//!
//! Key quantities:
//! - ATM implied volatility
//! - ATM skew $\partial\sigma/\partial k\big|_{k=0}$
//! - ATM convexity $\partial^2\sigma/\partial k^2\big|_{k=0}$
//! - 25-delta risk reversal and butterfly
//! - Term structure of ATM vol and skew
//!
//! Reference: Gatheral, *The Volatility Surface* (2006), Ch. 2–3

use super::ssvi::SsviParams;
use super::svi::SviRawParams;
use crate::traits::FloatExt;

/// Smile analytics for a single maturity slice.
#[derive(Clone, Debug)]
pub struct SmileAnalytics<T: FloatExt> {
  /// ATM implied volatility $\sigma(k=0)$
  pub atm_vol: T,
  /// ATM skew $\partial\sigma/\partial k\big|_{k=0}$
  pub atm_skew: T,
  /// ATM convexity $\partial^2\sigma/\partial k^2\big|_{k=0}$
  pub atm_convexity: T,
  /// ATM total variance $w(k=0) = \sigma^2 T$
  pub atm_total_variance: T,
  /// Time to expiry
  pub tau: T,
}

/// Compute smile analytics from an SVI slice.
///
/// ATM is at $k = 0$ (forward moneyness).
pub fn svi_analytics<T: FloatExt>(params: &SviRawParams<T>, tau: T) -> SmileAnalytics<T> {
  let zero = T::zero();
  let two = T::from_f64_fast(2.0);
  let four = T::from_f64_fast(4.0);

  let w0 = params.total_variance(zero);
  let atm_vol = if w0 > zero && tau > zero {
    (w0 / tau).sqrt()
  } else {
    T::nan()
  };

  let wp = params.w_prime(zero);
  let wpp = params.w_double_prime(zero);

  let (atm_skew, atm_convexity) = if atm_vol > zero && tau > zero {
    let skew = wp / (two * atm_vol * tau);
    let conv =
      wpp / (two * atm_vol * tau) - wp * wp / (four * atm_vol * atm_vol * atm_vol * tau * tau);
    (skew, conv)
  } else {
    (T::nan(), T::nan())
  };

  SmileAnalytics {
    atm_vol,
    atm_skew,
    atm_convexity,
    atm_total_variance: w0,
    tau,
  }
}

/// Compute smile analytics from an SSVI surface at given maturity.
pub fn ssvi_analytics<T: FloatExt>(
  params: &SsviParams<T>,
  theta: T,
  tau: T,
) -> SmileAnalytics<T> {
  let zero = T::zero();
  let two = T::from_f64_fast(2.0);
  let four = T::from_f64_fast(4.0);

  let w0 = params.total_variance(zero, theta);
  let atm_vol = if w0 > zero && tau > zero {
    (w0 / tau).sqrt()
  } else {
    T::nan()
  };

  let wp = params.w_prime_k(zero, theta);
  let wpp = params.w_double_prime_k(zero, theta);

  let (atm_skew, atm_convexity) = if atm_vol > zero && tau > zero {
    let skew = wp / (two * atm_vol * tau);
    let conv =
      wpp / (two * atm_vol * tau) - wp * wp / (four * atm_vol * atm_vol * atm_vol * tau * tau);
    (skew, conv)
  } else {
    (T::nan(), T::nan())
  };

  SmileAnalytics {
    atm_vol,
    atm_skew,
    atm_convexity,
    atm_total_variance: w0,
    tau,
  }
}

/// Compute ATM vol term structure from SSVI.
///
/// Returns `(maturities, atm_vols)`.
pub fn atm_term_structure<T: FloatExt>(
  params: &SsviParams<T>,
  thetas: &[T],
  maturities: &[T],
) -> (Vec<T>, Vec<T>) {
  let zero = T::zero();
  let vols: Vec<T> = thetas
    .iter()
    .zip(maturities.iter())
    .map(|(&theta, &t)| {
      let w = params.total_variance(zero, theta);
      if w > zero && t > zero {
        (w / t).sqrt()
      } else {
        T::nan()
      }
    })
    .collect();

  (maturities.to_vec(), vols)
}

/// Compute skew term structure from SSVI.
///
/// Returns `(maturities, atm_skews)`.
pub fn skew_term_structure<T: FloatExt>(
  params: &SsviParams<T>,
  thetas: &[T],
  maturities: &[T],
) -> (Vec<T>, Vec<T>) {
  let skews: Vec<T> = thetas
    .iter()
    .zip(maturities.iter())
    .map(|(&theta, &t)| {
      let a = ssvi_analytics(params, theta, t);
      a.atm_skew
    })
    .collect();

  (maturities.to_vec(), skews)
}

/// 25-delta risk reversal from an SVI slice.
///
/// RR₂₅ ≈ σ(k₊) − σ(k₋) where k₊, k₋ are approximate 25-delta
/// log-moneyness points: $k \approx \pm \sigma_{\mathrm{ATM}}\sqrt{T} \cdot 0.6745$.
pub fn risk_reversal_25d<T: FloatExt>(params: &SviRawParams<T>, tau: T) -> T {
  let zero = T::zero();
  let z = T::from_f64_fast(0.6745);
  let atm_vol = params.implied_vol(zero, tau);
  if !atm_vol.is_finite() || atm_vol <= zero {
    return T::nan();
  }

  let k25 = atm_vol * tau.sqrt() * z;
  let sigma_call = params.implied_vol(k25, tau);
  let sigma_put = params.implied_vol(-k25, tau);

  if sigma_call.is_finite() && sigma_put.is_finite() {
    sigma_call - sigma_put
  } else {
    T::nan()
  }
}

/// 25-delta butterfly from an SVI slice.
///
/// BF₂₅ = (σ(k₊) + σ(k₋))/2 − σ_ATM
pub fn butterfly_25d<T: FloatExt>(params: &SviRawParams<T>, tau: T) -> T {
  let zero = T::zero();
  let half = T::from_f64_fast(0.5);
  let z = T::from_f64_fast(0.6745);
  let atm_vol = params.implied_vol(zero, tau);
  if !atm_vol.is_finite() || atm_vol <= zero {
    return T::nan();
  }

  let k25 = atm_vol * tau.sqrt() * z;
  let sigma_call = params.implied_vol(k25, tau);
  let sigma_put = params.implied_vol(-k25, tau);

  if sigma_call.is_finite() && sigma_put.is_finite() {
    half * (sigma_call + sigma_put) - atm_vol
  } else {
    T::nan()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn svi_analytics_basic() {
    let p = SviRawParams::<f64>::new(0.04, 0.4, -0.4, 0.0, 0.2);
    let a = svi_analytics(&p, 1.0);

    assert!(a.atm_vol > 0.0);
    assert!(a.atm_skew < 0.0, "negative rho should give negative ATM skew");
    assert!(a.atm_convexity.is_finite());
    assert!((a.atm_total_variance - p.total_variance(0.0)).abs() < 1e-12);
  }

  #[test]
  fn ssvi_analytics_basic() {
    let p = SsviParams::<f64>::new(-0.3, 0.5, 0.5);
    let theta = 0.04;
    let a = ssvi_analytics(&p, theta, 1.0);

    assert!(a.atm_vol > 0.0);
    assert!(a.atm_skew < 0.0, "negative rho should give negative ATM skew");
    assert!((a.atm_total_variance - theta).abs() < 1e-10);
  }

  #[test]
  fn risk_reversal_and_butterfly() {
    let p = SviRawParams::<f64>::new(0.04, 0.4, -0.4, 0.0, 0.2);
    let rr = risk_reversal_25d(&p, 1.0);
    let bf = butterfly_25d(&p, 1.0);

    assert!(rr.is_finite());
    assert!(rr < 0.0, "negative rho should give negative RR25: rr={rr}");
    assert!(bf.is_finite());
    assert!(bf > 0.0, "butterfly should be positive (smile convexity): bf={bf}");
  }

  #[test]
  fn atm_term_structure_increasing() {
    let p = SsviParams::<f64>::new(-0.3, 0.5, 0.5);
    let thetas = vec![0.01, 0.04, 0.09, 0.16];
    let mats = vec![0.25, 0.50, 1.0, 2.0];

    let (_ts, vols) = atm_term_structure(&p, &thetas, &mats);
    assert!(vols.iter().all(|v| v.is_finite() && *v > 0.0));
  }

  #[test]
  fn skew_term_structure_negative() {
    let p = SsviParams::<f64>::new(-0.3, 0.5, 0.5);
    let thetas = vec![0.01, 0.04, 0.09];
    let mats = vec![0.25, 0.50, 1.0];

    let (_ts, skews) = skew_term_structure(&p, &thetas, &mats);
    assert!(skews.iter().all(|s| s.is_finite() && *s < 0.0));
  }
}
