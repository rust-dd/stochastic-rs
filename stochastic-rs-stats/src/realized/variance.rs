//! Realized variance, semivariance and higher realized moments.
//!
//! Reference: Andersen, Bollerslev, "Answering the Skeptics: Yes, Standard
//! Volatility Models Do Provide Accurate Forecasts", International Economic
//! Review, 39(4), 885-905 (1998). DOI: 10.2307/2527343
//!
//! Reference: Barndorff-Nielsen, Kinnebrock, Shephard, "Measuring Downside
//! Risk — Realised Semivariance" (2010). DOI:
//! 10.1093/acprof:oso/9780199549498.003.0007
//!
//! Reference: Amaya, Christoffersen, Jacobs, Vasquez, "Does Realized Skewness
//! Predict the Cross-Section of Equity Returns?", Journal of Financial
//! Economics, 118(1), 135-167 (2015). DOI: 10.1016/j.jfineco.2015.02.009

use ndarray::Array1;
use ndarray::ArrayView1;

use crate::traits::FloatExt;

/// Log-returns $r_i = \ln(p_i / p_{i-1})$ of a price path, empty below two prices; a non-positive
/// price gives NaN returns on both sides.
pub fn log_returns<T: FloatExt>(prices: ArrayView1<T>) -> Array1<T> {
  let n = prices.len().saturating_sub(1);
  let mut out = Array1::<T>::zeros(n);
  for i in 0..n {
    let p0 = prices[i];
    let p1 = prices[i + 1];
    out[i] = if p0 > T::zero() && p1 > T::zero() {
      (p1 / p0).ln()
    } else {
      T::nan()
    };
  }
  out
}

/// Realized variance $RV = \sum_i r_i^2$ of intraday log-returns (see [`log_returns`]); NaN for no
/// returns.
pub fn realized_variance<T: FloatExt>(returns: ArrayView1<T>) -> T {
  if returns.is_empty() {
    return T::nan();
  }
  returns.iter().fold(T::zero(), |acc, &r| acc + r * r)
}

/// Realized volatility $\sqrt{a \cdot RV}$ for an annualisation factor $a$ (`1.0` for the raw
/// realized standard deviation); NaN for no returns.
pub fn realized_volatility<T: FloatExt>(returns: ArrayView1<T>, annualisation: T) -> T {
  (annualisation * realized_variance(returns)).sqrt()
}

/// Realized semivariances `(RV⁻, RV⁺)` below and above `threshold` (typically `0`), a return equal
/// to it in neither; NaN for no returns or a NaN one, and panics unless `threshold` is finite.
pub fn realized_semivariance<T: FloatExt>(returns: ArrayView1<T>, threshold: T) -> (T, T) {
  assert!(
    threshold.is_finite(),
    "threshold must satisfy `threshold.is_finite()`, got threshold = {threshold:?}"
  );
  if returns.is_empty() {
    return (T::nan(), T::nan());
  }
  let mut down = T::zero();
  let mut up = T::zero();
  for &r in returns.iter() {
    let r2 = r * r;
    if r < threshold {
      down += r2;
    } else if r > threshold {
      up += r2;
    } else if r.is_nan() {
      return (T::nan(), T::nan());
    }
  }
  (down, up)
}

/// Realized skewness $RSK = \sqrt n \sum_i r_i^3 / RV^{3/2}$ (Amaya et al. 2015); NaN below two
/// returns or for zero $RV$.
pub fn realized_skewness<T: FloatExt>(returns: ArrayView1<T>) -> T {
  let n = returns.len();
  if n < 2 {
    return T::nan();
  }
  let rv = realized_variance(returns);
  if rv == T::zero() {
    return T::nan();
  }
  let cube_sum = returns.iter().fold(T::zero(), |acc, &r| acc + r * r * r);
  T::from_usize_(n).sqrt() * cube_sum / rv.powf(T::from_f64_fast(1.5))
}

/// Realized kurtosis $RKT = n \sum_i r_i^4 / RV^2$ (Amaya et al. 2015); NaN below two returns or
/// for zero $RV$.
pub fn realized_kurtosis<T: FloatExt>(returns: ArrayView1<T>) -> T {
  let n = returns.len();
  if n < 2 {
    return T::nan();
  }
  let rv = realized_variance(returns);
  if rv == T::zero() {
    return T::nan();
  }
  let quad_sum = returns.iter().fold(T::zero(), |acc, &r| {
    let r2 = r * r;
    acc + r2 * r2
  });
  T::from_usize_(n) * quad_sum / (rv * rv)
}

/// Realized quarticity $RQ = \frac{n}{3} \sum_i r_i^4$, the no-jump variance estimator of $RV$
/// (BN-Shephard 2002); NaN for no returns.
pub fn realized_quarticity<T: FloatExt>(returns: ArrayView1<T>) -> T {
  let n = returns.len();
  if n == 0 {
    return T::nan();
  }
  let quad_sum = returns.iter().fold(T::zero(), |acc, &r| {
    let r2 = r * r;
    acc + r2 * r2
  });
  T::from_usize_(n) / T::from_f64_fast(3.0) * quad_sum
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  #[test]
  fn log_returns_matches_hand_computation() {
    let p = array![100.0_f64, 101.0, 100.5, 102.0];
    let r = log_returns(p.view());
    assert_eq!(r.len(), 3);
    assert!(approx(r[0], (101.0_f64 / 100.0).ln(), 1e-12));
    assert!(approx(r[1], (100.5_f64 / 101.0).ln(), 1e-12));
    assert!(approx(r[2], (102.0_f64 / 100.5).ln(), 1e-12));
  }

  #[test]
  fn fewer_than_two_prices_give_no_returns() {
    assert!(log_returns(Array1::<f64>::zeros(0).view()).is_empty());
    assert!(log_returns(array![100.0_f64].view()).is_empty());
  }

  #[test]
  fn a_non_positive_price_gives_a_nan_return() {
    let r = log_returns(array![100.0_f64, 0.0, 101.0, -1.0, 102.0].view());
    assert!(r.iter().all(|v| v.is_nan()), "{r:?}");
  }

  #[test]
  fn rv_constant_returns_equals_n_times_r_squared() {
    let r = Array1::<f64>::from_elem(100, 0.01);
    let rv = realized_variance(r.view());
    assert!(approx(rv, 100.0 * 0.0001, 1e-12));
  }

  #[test]
  fn semivariance_decomposes_rv() {
    let r = array![-0.02_f64, 0.01, 0.0, 0.03, -0.005];
    let (rs_minus, rs_plus) = realized_semivariance(r.view(), 0.0);
    let rv = realized_variance(r.view());
    assert!(approx(rs_minus + rs_plus, rv, 1e-15));
    assert!(approx(
      rs_minus,
      0.02_f64.powi(2) + 0.005_f64.powi(2),
      1e-15
    ));
    assert!(approx(rs_plus, 0.01_f64.powi(2) + 0.03_f64.powi(2), 1e-15));
  }

  #[test]
  #[should_panic(expected = "threshold must satisfy `threshold.is_finite()`, got threshold = NaN")]
  fn a_nan_threshold_panics() {
    let _ = realized_semivariance(array![0.01_f64, -0.02].view(), f64::NAN);
  }

  #[test]
  fn skewness_zero_for_symmetric_sample() {
    let r = array![-0.02_f64, 0.02, -0.01, 0.01];
    let s = realized_skewness(r.view());
    assert!(s.abs() < 1e-12);
  }

  #[test]
  fn kurtosis_of_constant_returns_equals_one() {
    // For constant r_i = c: RKT = n·sum r_i^4 / RV^2 = n·n·c^4 / (n·c^2)^2 = 1.
    let r = Array1::<f64>::from_elem(50, 0.01);
    let k = realized_kurtosis(r.view());
    assert!(approx(k, 1.0, 1e-9));
  }
}
