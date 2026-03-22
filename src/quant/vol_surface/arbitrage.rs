//! # Arbitrage-free conditions for implied volatility surfaces
//!
//! Butterfly arbitrage density function (Gatheral & Jacquier, 2012, §2):
//!
//! $$
//! g(k)=\Bigl(1-\frac{k\,w'(k)}{2\,w(k)}\Bigr)^2
//!     -\frac{w'(k)^2}{4}\Bigl(\frac{1}{w(k)}+\frac{1}{4}\Bigr)
//!     +\frac{w''(k)}{2}
//! $$
//!
//! A slice is free of butterfly arbitrage iff $g(k)\geq0$ for all $k$.
//!
//! Calendar-spread arbitrage is absent iff $\partial_t w(k,t)\geq0$ for all $k$.
//!
//! Roger Lee's moment formula (Lee, 2004) gives the asymptotic bound:
//! $\limsup_{k\to\pm\infty} w(k)/|k| \leq 2$.
//!
//! Reference: Gatheral & Jacquier (2012), arXiv:1204.0646
//! Reference: Lee (2004), "The Moment Formula for Implied Volatility at Extreme Strikes"

use super::ssvi::SsviParams;
use super::svi::SviRawParams;
use crate::traits::FloatExt;

/// Butterfly arbitrage density $g(k)$ at point $k$.
///
/// $$
/// g(k) = \Bigl(1 - \frac{k\,w'(k)}{2\,w(k)}\Bigr)^2
///     - \frac{w'(k)^2}{4}\Bigl(\frac{1}{w(k)} + \frac{1}{4}\Bigr)
///     + \frac{w''(k)}{2}
/// $$
#[inline]
pub fn butterfly_density_at<T: FloatExt>(k: T, w: T, w_prime: T, w_double_prime: T) -> T {
  if w <= T::zero() {
    return T::nan();
  }
  let half = T::from_f64_fast(0.5);
  let quarter = T::from_f64_fast(0.25);
  let two = T::from_f64_fast(2.0);
  let one = T::one();

  let p = one - k * w_prime / (two * w);
  p * p - quarter * w_prime * w_prime * (one / w + quarter) + half * w_double_prime
}

/// Check butterfly arbitrage for an SVI slice on a grid of log-moneyness values.
///
/// Returns `(is_free, min_g)` where `min_g` is the minimum density value.
pub fn check_butterfly_svi<T: FloatExt>(params: &SviRawParams<T>, ks: &[T]) -> (bool, T) {
  let mut min_g = T::infinity();

  for &k in ks {
    let w = params.total_variance(k);
    let wp = params.w_prime(k);
    let wpp = params.w_double_prime(k);
    let g = butterfly_density_at(k, w, wp, wpp);

    if g.is_finite() && g < min_g {
      min_g = g;
    }
  }

  (min_g >= T::zero(), min_g)
}

/// Check butterfly arbitrage for an SSVI slice at given $\theta$.
pub fn check_butterfly_ssvi<T: FloatExt>(
  params: &SsviParams<T>,
  theta: T,
  ks: &[T],
) -> (bool, T) {
  let mut min_g = T::infinity();

  for &k in ks {
    let w = params.total_variance(k, theta);
    let wp = params.w_prime_k(k, theta);
    let wpp = params.w_double_prime_k(k, theta);
    let g = butterfly_density_at(k, w, wp, wpp);

    if g.is_finite() && g < min_g {
      min_g = g;
    }
  }

  (min_g >= T::zero(), min_g)
}

/// Check calendar-spread arbitrage: $\partial_t w(k,t) \geq 0$ for all
/// grid points.
///
/// Returns `(is_free, worst_violation)`.
pub fn check_calendar_spread(
  total_variance: &ndarray::Array2<f64>,
  maturities: &[f64],
) -> (bool, f64) {
  let nt = maturities.len();
  let nk = total_variance.ncols();
  let mut worst = f64::INFINITY;

  for j in 1..nt {
    let dt = maturities[j] - maturities[j - 1];
    if dt <= 0.0 {
      continue;
    }
    for i in 0..nk {
      let dw = total_variance[[j, i]] - total_variance[[j - 1, i]];
      if dw.is_finite() && dw < worst {
        worst = dw;
      }
    }
  }

  (worst >= 0.0, worst)
}

/// Roger Lee moment formula asymptotic slope bounds.
///
/// Returns `(right_slope, left_slope)` estimated from the outermost points.
pub fn lee_moment_slopes<T: FloatExt>(ks: &[T], ws: &[T]) -> (T, T) {
  assert!(ks.len() >= 2);
  let n = ks.len();
  let zero = T::zero();

  let right_slope = if ks[n - 1] > zero {
    ws[n - 1] / ks[n - 1]
  } else {
    zero
  };

  let left_slope = if ks[0] < zero {
    ws[0] / ks[0].abs()
  } else {
    zero
  };

  (right_slope, left_slope)
}

/// Check whether wing slopes satisfy Roger Lee bounds ($\leq 2$).
pub fn check_lee_bounds<T: FloatExt>(ks: &[T], ws: &[T]) -> bool {
  let (right, left) = lee_moment_slopes(ks, ws);
  let two = T::from_f64_fast(2.0);
  let eps = T::from_f64_fast(1e-10);
  right <= two + eps && left <= two + eps
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn butterfly_admissible_svi() {
    let p = SviRawParams::<f64>::new(0.04, 0.2, -0.3, 0.0, 0.3);
    let ks: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.05).collect();
    let (free, min_g) = check_butterfly_svi(&p, &ks);
    assert!(free, "admissible SVI should be butterfly-free, min_g={min_g}");
  }

  #[test]
  fn butterfly_admissible_ssvi() {
    let p = SsviParams::<f64>::new(-0.3, 0.5, 0.5);
    assert!(p.satisfies_no_butterfly_condition());

    let ks: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.05).collect();
    let (free, min_g) = check_butterfly_ssvi(&p, 0.04, &ks);
    assert!(free, "SSVI with η(1+|ρ|)≤2 should be butterfly-free, min_g={min_g}");
  }

  #[test]
  fn lee_bounds_svi() {
    let p = SviRawParams::<f64>::new(0.04, 0.2, -0.3, 0.0, 0.3);
    let ks: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.1).collect();
    let ws: Vec<f64> = ks.iter().map(|&k| p.total_variance(k)).collect();
    assert!(check_lee_bounds(&ks, &ws));
  }

  #[test]
  fn calendar_spread_increasing_theta() {
    let p = SsviParams::<f64>::new(-0.3, 0.5, 0.5);
    let thetas = [0.02, 0.04, 0.08];
    let ks: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.1).collect();

    let mut surface = ndarray::Array2::<f64>::zeros((thetas.len(), ks.len()));
    for (j, &theta) in thetas.iter().enumerate() {
      for (i, &k) in ks.iter().enumerate() {
        surface[[j, i]] = p.total_variance(k, theta);
      }
    }

    let maturities = [0.25, 0.50, 1.0];
    let (free, _worst) = check_calendar_spread(&surface, &maturities);
    assert!(free, "increasing thetas should be calendar-spread free");
  }
}
