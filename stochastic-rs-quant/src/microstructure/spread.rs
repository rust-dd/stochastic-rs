//! Bid-ask spread estimators from trade and quote data.
//!
//! - **Roll (1984)**: implicit effective spread from the negative serial
//!   covariance of trade-price changes induced by the bid-ask bounce.
//! - **Effective spread**: realised cost from trade prices and contemporaneous
//!   mid-quotes — $2|p_t - m_t|$ for buys and equivalent for sells.
//! - **Corwin-Schultz (2012)**: 2-day high-low spread estimator from daily
//!   highs and lows alone.
//!
//! Reference: Roll, "A Simple Implicit Measure of the Effective Bid-Ask
//! Spread in an Efficient Market", Journal of Finance, 39(4), 1127-1139
//! (1984). DOI: 10.1111/j.1540-6261.1984.tb03897.x
//!
//! Reference: Corwin, Schultz, "A Simple Way to Estimate Bid-Ask Spreads from
//! Daily High and Low Prices", Journal of Finance, 67(2), 719-760 (2012).
//! DOI: 10.1111/j.1540-6261.2012.01729.x

use ndarray::Array1;
use ndarray::ArrayView1;

use crate::traits::FloatExt;

/// Roll (1984) implicit effective spread:
/// $s_{Roll} = 2\sqrt{\max(0, -\widehat{\mathrm{Cov}}(\Delta p_t, \Delta p_{t-1}))}$.
///
/// Estimator divides by the **number of cross-product terms** $n$ (i.e. the
/// number of $(\Delta p_t, \Delta p_{t-1})$ pairs, which equals
/// `diffs.len() - 1` for a length-$n$ price series). This matches Roll
/// (1984)'s original derivation. The alternative $(n-1)$ divisor (sample
/// autocovariance) is also defensible — both estimators are consistent and
/// differ by $O(1/n)$ on long series. The current implementation uses the
/// Roll-1984 convention.
pub fn roll_spread<T: FloatExt>(prices: ArrayView1<T>) -> T {
  let n = prices.len();
  if n < 3 {
    return T::zero();
  }
  let mut diffs = Array1::<T>::zeros(n - 1);
  for i in 0..(n - 1) {
    diffs[i] = prices[i + 1] - prices[i];
  }
  let mean = diffs.iter().fold(T::zero(), |a, &v| a + v) / T::from_usize_(diffs.len());
  let mut cov = T::zero();
  let mut count = 0usize;
  for i in 1..diffs.len() {
    cov += (diffs[i] - mean) * (diffs[i - 1] - mean);
    count += 1;
  }
  if count == 0 {
    return T::zero();
  }
  cov = cov / T::from_usize_(count);
  if cov >= T::zero() {
    return T::zero();
  }
  T::from_f64_fast(2.0) * (-cov).sqrt()
}

/// Realised effective half-spread from contemporaneous trades and mids.
///
/// `trade_price[i]` is the executed trade price at tick `i` and `mid[i]`
/// the prevailing mid-quote at the same instant. Returns the average
/// $\frac{1}{N}\sum_i 2\,|p_i - m_i|$ — the realised round-trip cost.
pub fn effective_spread<T: FloatExt>(trade_price: ArrayView1<T>, mid: ArrayView1<T>) -> T {
  assert_eq!(
    trade_price.len(),
    mid.len(),
    "trade-price and mid series must have the same length"
  );
  let n = trade_price.len();
  if n == 0 {
    return T::zero();
  }
  let mut acc = T::zero();
  for i in 0..n {
    let d = trade_price[i] - mid[i];
    let absd = if d < T::zero() { -d } else { d };
    acc += T::from_f64_fast(2.0) * absd;
  }
  acc / T::from_usize_(n)
}

/// Corwin-Schultz (2012) 2-day high-low bid-ask spread estimator.
///
/// `high[i]`/`low[i]` are the daily high and low prices for day `i`. Returns
/// the average 2-day spread $\bar S$ across all consecutive day pairs:
///
/// $$
/// S = \frac{2(e^\alpha-1)}{1+e^\alpha},
/// \quad
/// \alpha = \frac{\sqrt{2\beta}-\sqrt{\beta}}{3-2\sqrt 2} - \sqrt{\frac{\gamma}{3-2\sqrt 2}},
/// $$
///
/// with $\beta = \mathbb E[(\ln(H_t/L_t))^2 + (\ln(H_{t+1}/L_{t+1}))^2]$ and
/// $\gamma = \mathbb E[(\ln(H_t^{(2)}/L_t^{(2)}))^2]$ where the
/// superscript $(2)$ denotes the 2-day high/low.
pub fn corwin_schultz_spread<T: FloatExt>(high: ArrayView1<T>, low: ArrayView1<T>) -> T {
  assert_eq!(high.len(), low.len(), "high and low must have equal length");
  let n = high.len();
  if n < 2 {
    return T::zero();
  }
  let three_minus_2sqrt2 = T::from_f64_fast(3.0 - 2.0 * std::f64::consts::SQRT_2);
  let two = T::from_f64_fast(2.0);
  let mut acc = T::zero();
  let mut count = 0usize;
  for t in 0..(n - 1) {
    let h_t = high[t];
    let l_t = low[t];
    let h_t1 = high[t + 1];
    let l_t1 = low[t + 1];
    if h_t <= T::zero() || l_t <= T::zero() || h_t1 <= T::zero() || l_t1 <= T::zero() {
      continue;
    }
    let ln_hl_t = (h_t / l_t).ln();
    let ln_hl_t1 = (h_t1 / l_t1).ln();
    let beta = ln_hl_t * ln_hl_t + ln_hl_t1 * ln_hl_t1;
    let h2 = if h_t > h_t1 { h_t } else { h_t1 };
    let l2 = if l_t < l_t1 { l_t } else { l_t1 };
    if h2 <= T::zero() || l2 <= T::zero() {
      continue;
    }
    let ln_hl_2 = (h2 / l2).ln();
    let gamma = ln_hl_2 * ln_hl_2;
    let alpha = ((two * beta).sqrt() - beta.sqrt()) / three_minus_2sqrt2
      - (gamma / three_minus_2sqrt2).sqrt();
    let exp_alpha = alpha.exp();
    let s = two * (exp_alpha - T::one()) / (T::one() + exp_alpha);
    acc += s;
    count += 1;
  }
  if count == 0 {
    T::zero()
  } else {
    acc / T::from_usize_(count)
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  #[test]
  fn roll_zero_when_no_bounce() {
    let p = array![100.0_f64, 100.5, 101.0, 101.5, 102.0];
    assert!(approx(roll_spread(p.view()), 0.0, 1e-12));
  }

  #[test]
  fn roll_recovers_known_spread_under_pure_bounce() {
    // Roll's estimator has standard error roughly s/sqrt(2n) under the
    // bid/ask bounce DGP. For n = 10_000 and s = 0.10 that is ≈ 7e-4, so
    // a 0.03 tolerance is ≈ 40σ — comfortable. We tested seeds 0, 7, 11,
    // 42, and 100 against this tolerance; all pass with margin > 25σ.
    // If you tighten the tolerance below 0.005 you must enlarge n.
    let mid = 100.0_f64;
    let s = 0.10;
    let n = 10_000;
    let buy_sell = SimdNormal::<f64>::with_seed(0.0, 1.0, 11);
    let mut signs = vec![0.0_f64; n];
    buy_sell.fill_slice_fast(&mut signs);
    let p = Array1::from_iter(signs.iter().map(|&z| {
      let sign = if z >= 0.0 { 1.0 } else { -1.0 };
      mid + 0.5 * s * sign
    }));
    let est = roll_spread(p.view());
    assert!(approx(est, s, 0.03));
  }

  #[test]
  fn effective_spread_matches_hand_average() {
    let p = array![100.05_f64, 99.95, 100.10, 99.90];
    let m = array![100.00_f64, 100.00, 100.00, 100.00];
    let est = effective_spread(p.view(), m.view());
    assert!(approx(est, (0.10 + 0.10 + 0.20 + 0.20) / 4.0, 1e-12));
  }

  #[test]
  fn corwin_schultz_finite_for_simple_path() {
    let h = array![101.0_f64, 102.0, 103.0, 102.5];
    let l = array![100.0_f64, 100.5, 101.0, 101.0];
    let s = corwin_schultz_spread(h.view(), l.view());
    assert!(s.is_finite());
  }

  #[test]
  fn corwin_schultz_zero_when_high_equals_low() {
    let h = array![100.0_f64, 100.0, 100.0];
    let l = array![100.0_f64, 100.0, 100.0];
    let s = corwin_schultz_spread(h.view(), l.view());
    assert!(approx(s, 0.0, 1e-12));
  }
}
