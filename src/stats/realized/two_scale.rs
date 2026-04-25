//! Two-Scale and Multi-Scale Realized Variance estimators.
//!
//! Decomposes the standard realized variance bias under additive noise into a
//! "slow-scale" component obtained by sub-sampling at a coarser frequency, and
//! a "fast-scale" RV obtained from all observations. Combining the two cancels
//! the leading-order noise bias term.
//!
//! $$
//! \widehat{TSRV} = \widehat{RV}^{(K)} - \frac{\bar n_K}{n}\widehat{RV}^{(\mathrm{all})},
//! \qquad \bar n_K = \frac{n-K+1}{K}.
//! $$
//!
//! The multi-scale variant (Zhang 2006) combines $M$ sub-sampling frequencies
//! using $a_i = \lambda + \mu/i$ where $(\lambda, \mu)$ enforce the two
//! orthogonality constraints $\sum a_i = 1$ (consistency) and
//! $\sum a_i/i = 0$ (noise-bias cancellation), yielding the closed form
//! $\mu = H_M / (H_M^2 - M H_M^{(2)})$, $\lambda = -H_M^{(2)} \mu / H_M$ with
//! $H_M = \sum_{i=1}^M 1/i$ and $H_M^{(2)} = \sum_{i=1}^M 1/i^2$.
//!
//! Reference: Zhang, Mykland, Aït-Sahalia, "A Tale of Two Time Scales:
//! Determining Integrated Volatility With Noisy High-Frequency Data", Journal
//! of the American Statistical Association, 100(472), 1394-1411 (2005).
//! DOI: 10.1198/016214505000000169
//!
//! Reference: Zhang, "Efficient Estimation of Stochastic Volatility Using Noisy
//! Observations: A Multi-Scale Approach", Bernoulli, 12(6), 1019-1043 (2006).
//! DOI: 10.3150/bj/1165269149

use ndarray::ArrayView1;

use crate::traits::FloatExt;

/// Two-Scale Realized Variance (Zhang, Mykland, Aït-Sahalia 2005).
///
/// `prices` is the (length $n+1$) intraday price path. `k` is the
/// sub-sampling factor — the slow scale uses every $k$-th price. Typical
/// values are 5, 10, 20, or chosen via $K^{*} = c\,n^{2/3}$.
pub fn two_scale_rv<T: FloatExt>(prices: ArrayView1<T>, k: usize) -> T {
  let len = prices.len();
  assert!(k >= 1, "subsampling factor must be ≥ 1");
  assert!(len >= k + 2, "price path too short for k = {k}");
  let n = len - 1;
  let rv_all = price_rv(prices, 1);
  let rv_slow = average_subsample_rv(prices, k);
  let nbar = T::from_f64_fast((n as f64 - k as f64 + 1.0) / k as f64);
  let nf = T::from_usize_(n);
  let factor = T::one() - nbar / nf;
  let raw = rv_slow - (nbar / nf) * rv_all;
  if factor > T::zero() {
    raw / factor
  } else {
    raw
  }
}

/// Multi-Scale Realized Variance (Zhang 2006).
///
/// Combines RVs at scales $1, 2, \ldots, M$ with weights of the form
/// $a_i = \lambda + \mu/i$, where $(\lambda, \mu)$ are chosen so that
/// $\sum_i a_i = 1$ (consistency) and $\sum_i a_i/i = 0$ (cancellation of
/// the leading-order noise bias). The closed form is
/// $\mu = H_M / (H_M^2 - M H_M^{(2)})$ and
/// $\lambda = -H_M^{(2)}\,\mu / H_M$ with $H_M = \sum 1/i$,
/// $H_M^{(2)} = \sum 1/i^2$.
pub fn multi_scale_rv<T: FloatExt>(prices: ArrayView1<T>, scales: usize) -> T {
  let m = scales.max(2);
  assert!(prices.len() >= m + 2, "price path too short for {m} scales");
  let h1: f64 = (1..=m).map(|i| 1.0 / i as f64).sum();
  let h2: f64 = (1..=m).map(|i| 1.0 / (i as f64).powi(2)).sum();
  let mu = h1 / (h1.powi(2) - (m as f64) * h2);
  let lambda = -h2 * mu / h1;
  let mut acc = T::zero();
  for i in 1..=m {
    let weight = lambda + mu / i as f64;
    let rv_i = average_subsample_rv(prices, i);
    acc += T::from_f64_fast(weight) * rv_i;
  }
  acc
}

fn price_rv<T: FloatExt>(prices: ArrayView1<T>, step: usize) -> T {
  let mut acc = T::zero();
  let mut i = step;
  while i < prices.len() {
    let d = prices[i] - prices[i - step];
    acc += d * d;
    i += step;
  }
  acc
}

fn average_subsample_rv<T: FloatExt>(prices: ArrayView1<T>, k: usize) -> T {
  if k <= 1 {
    return price_rv(prices, 1);
  }
  let mut total = T::zero();
  let mut count = 0usize;
  for offset in 0..k {
    let mut acc = T::zero();
    let mut i = offset + k;
    while i < prices.len() {
      let d = prices[i] - prices[i - k];
      acc += d * d;
      i += k;
    }
    total += acc;
    count += 1;
  }
  total / T::from_usize_(count)
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;

  use super::*;
  use crate::distributions::normal::SimdNormal;

  fn simulate_noisy_path(seed: u64, n: usize, sigma: f64, omega: f64) -> (Array1<f64>, f64) {
    let dx = SimdNormal::<f64>::with_seed(0.0, sigma, seed);
    let dn = SimdNormal::<f64>::with_seed(0.0, omega, seed.wrapping_add(1));
    let mut steps = vec![0.0_f64; n];
    dx.fill_slice_fast(&mut steps);
    let mut noise = vec![0.0_f64; n + 1];
    dn.fill_slice_fast(&mut noise);
    let mut x = vec![0.0_f64; n + 1];
    for i in 1..=n {
      x[i] = x[i - 1] + steps[i - 1];
    }
    let y: Vec<f64> = x
      .iter()
      .zip(noise.iter())
      .map(|(&xv, &nv)| xv + nv)
      .collect();
    let iv = (n as f64) * sigma.powi(2);
    (Array1::from(y), iv)
  }

  #[test]
  fn tsrv_corrects_naive_rv_bias() {
    let (y, iv) = simulate_noisy_path(211, 20_000, 0.005, 0.003);
    let dy = Array1::from_iter((1..y.len()).map(|i| y[i] - y[i - 1]));
    let rv: f64 = dy.iter().map(|v| v * v).sum();
    let tsrv = two_scale_rv(y.view(), 20);
    assert!((tsrv - iv).abs() < (rv - iv).abs());
  }

  #[test]
  fn msrv_corrects_naive_rv_bias() {
    let (y, iv) = simulate_noisy_path(229, 20_000, 0.005, 0.003);
    let dy = Array1::from_iter((1..y.len()).map(|i| y[i] - y[i - 1]));
    let rv: f64 = dy.iter().map(|v| v * v).sum();
    let msrv = multi_scale_rv(y.view(), 12);
    assert!((msrv - iv).abs() < (rv - iv).abs());
  }
}
