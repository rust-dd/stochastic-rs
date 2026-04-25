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

/// Convert a price path to log-returns: $r_i = \ln(p_i / p_{i-1})$.
pub fn log_returns<T: FloatExt>(prices: ArrayView1<T>) -> Array1<T> {
  assert!(prices.len() >= 2, "need at least two prices");
  let n = prices.len() - 1;
  let mut out = Array1::<T>::zeros(n);
  for i in 0..n {
    let p0 = prices[i];
    let p1 = prices[i + 1];
    assert!(
      p0 > T::zero() && p1 > T::zero(),
      "prices must be strictly positive"
    );
    out[i] = (p1 / p0).ln();
  }
  out
}

/// Realized variance: $RV = \sum_i r_i^2$.
///
/// Pass intraday log-returns (use [`log_returns`] to transform a price path).
pub fn realized_variance<T: FloatExt>(returns: ArrayView1<T>) -> T {
  returns.iter().fold(T::zero(), |acc, &r| acc + r * r)
}

/// Realized volatility: $\sqrt{a \cdot RV}$ where $a$ is the user-supplied
/// annualisation factor (e.g. number of trading days per year if `returns` is
/// the daily realized variance, or `1.0` for raw realized standard deviation).
pub fn realized_volatility<T: FloatExt>(returns: ArrayView1<T>, annualisation: T) -> T {
  (annualisation * realized_variance(returns)).sqrt()
}

/// Realized semivariance with user-defined sign threshold (typically `0`).
///
/// Returns `(downside, upside)` such that
/// $RV^- = \sum r_i^2 \mathbf 1\{r_i < \tau\}$,
/// $RV^+ = \sum r_i^2 \mathbf 1\{r_i > \tau\}$
/// and $RV^- + RV^+ + \sum r_i^2 \mathbf 1\{r_i = \tau\} = RV$.
pub fn realized_semivariance<T: FloatExt>(returns: ArrayView1<T>, threshold: T) -> (T, T) {
  let mut down = T::zero();
  let mut up = T::zero();
  for &r in returns.iter() {
    let r2 = r * r;
    if r < threshold {
      down += r2;
    } else if r > threshold {
      up += r2;
    }
  }
  (down, up)
}

/// Realized skewness (Amaya et al. 2015):
/// $RSK = \frac{\sqrt n \sum_i r_i^3}{RV^{3/2}}$.
pub fn realized_skewness<T: FloatExt>(returns: ArrayView1<T>) -> T {
  let n = returns.len();
  assert!(n >= 2, "need at least two returns for skewness");
  let rv = realized_variance(returns);
  if rv <= T::zero() {
    return T::zero();
  }
  let cube_sum = returns
    .iter()
    .fold(T::zero(), |acc, &r| acc + r * r * r);
  T::from_usize_(n).sqrt() * cube_sum / rv.powf(T::from_f64_fast(1.5))
}

/// Realized kurtosis (Amaya et al. 2015):
/// $RKT = \frac{n \sum_i r_i^4}{RV^2}$.
pub fn realized_kurtosis<T: FloatExt>(returns: ArrayView1<T>) -> T {
  let n = returns.len();
  assert!(n >= 2, "need at least two returns for kurtosis");
  let rv = realized_variance(returns);
  if rv <= T::zero() {
    return T::zero();
  }
  let quad_sum = returns.iter().fold(T::zero(), |acc, &r| {
    let r2 = r * r;
    acc + r2 * r2
  });
  T::from_usize_(n) * quad_sum / (rv * rv)
}

/// Realized quarticity: $RQ = \frac{n}{3} \sum_i r_i^4$, the natural
/// $L^4$ companion to $RV$ used as the variance estimator of $RV$ under
/// no-jump asymptotics (BN-Shephard 2002).
pub fn realized_quarticity<T: FloatExt>(returns: ArrayView1<T>) -> T {
  let n = returns.len();
  let quad_sum = returns.iter().fold(T::zero(), |acc, &r| {
    let r2 = r * r;
    acc + r2 * r2
  });
  T::from_usize_(n) / T::from_f64_fast(3.0) * quad_sum
}

#[cfg(test)]
mod tests {
  use super::*;
  use ndarray::array;

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
    assert!(approx(rs_minus, 0.02_f64.powi(2) + 0.005_f64.powi(2), 1e-15));
    assert!(approx(rs_plus, 0.01_f64.powi(2) + 0.03_f64.powi(2), 1e-15));
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
