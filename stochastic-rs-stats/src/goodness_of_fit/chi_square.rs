//! Pearson chi-square goodness-of-fit test of binned counts against a fully specified pmf.
//!
//! $$
//! \chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}, \qquad E_i = n \cdot p_i, \qquad
//! \chi^2 \overset{H_0}{\sim} \chi^2_{k-1}
//! $$
//!
//! Nothing is fitted, so `k − 1` degrees of freedom; an expected count under 1 is an `Err`.
//!
//! Reference: Pearson (1900), Philosophical Magazine Series 5, 50(302), 157-175
//! Reference: Cochran (1954), Biometrics 10(4), 417-451

use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::input::assert_alpha;
use crate::traits::HypothesisTest;

/// Configuration for [`chi_square_gof_test`].
#[derive(Debug, Clone, Copy)]
pub struct ChiSquareGofConfig {
  /// Significance level. Default `0.05`.
  pub alpha: f64,
}

impl Default for ChiSquareGofConfig {
  fn default() -> Self {
    Self { alpha: 0.05 }
  }
}

/// Pearson chi-square goodness-of-fit test (null: the given pmf), decided by its p-value.
#[derive(Debug, Clone)]
pub struct ChiSquareGofTest {
  /// The `\chi^2` statistic.
  pub statistic: f64,
  /// p-value: `P(\chi^2_{df} > statistic)`.
  pub p_value: Option<f64>,
  /// The upper-tail `\chi^2_{df}` critical value at the `alpha` the test was run with.
  pub critical_values: Option<CriticalValues>,
  /// Whether the null is rejected at `alpha`.
  pub rejects_null: bool,
  /// Degrees of freedom, `bins - 1` (Pearson 1900; see module doc).
  pub df: usize,
}

impl HypothesisTest for ChiSquareGofTest {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.rejects_null)
  }
}

/// Upper-tail chi-square critical value: the `c` with `P(X > c) = alpha` for `X ~ ChiSquared(df)`
/// (`SimdChiSquared::quantile(1 - alpha)`); panics unless `0 < alpha < 1` and `df >= 1`.
pub fn chi_square_critical_value(alpha: f64, df: usize) -> f64 {
  assert!(alpha > 0.0 && alpha < 1.0, "alpha must be in (0, 1)");
  assert!(
    df >= 1,
    "chi-square test needs at least 1 degree of freedom"
  );
  use stochastic_rs_distributions::DistributionExt;
  use stochastic_rs_distributions::chi_square::SimdChiSquared;
  SimdChiSquared::<f64>::new(df as f64)
    .quantile(1.0 - alpha)
    .expect("SimdChiSquared has a closed-form quantile")
}

/// Pearson's chi-square test of counts against a fully specified null's masses; `Err` below 2 bins
/// or an expected count below 1; panics unless lengths match, masses sum to 1 and `0 < alpha < 1`.
pub fn chi_square_gof_test(
  observed: &[u64],
  expected_prob: &[f64],
  cfg: ChiSquareGofConfig,
) -> Result<ChiSquareGofTest, StatsError> {
  assert!(
    expected_prob.len() == observed.len(),
    "expected_prob must satisfy `expected_prob.len() == observed.len()`, got expected_prob.len() = {}, observed.len() = {}",
    expected_prob.len(),
    observed.len()
  );
  assert_alpha(cfg.alpha);
  if observed.len() < 2 {
    return Err(StatsError::InsufficientData {
      needed: 2,
      got: observed.len(),
    });
  }

  let n = observed.iter().sum::<u64>();
  let n_f = n as f64;
  let prob_sum = expected_prob.iter().sum::<f64>();
  assert!(
    (prob_sum - 1.0).abs() < 1e-6,
    "expected_prob must satisfy `(expected_prob.iter().sum() - 1.0).abs() < 1e-6`, got expected_prob.iter().sum() = {prob_sum:?}"
  );

  let mut statistic = 0.0_f64;
  for (&o, &p) in observed.iter().zip(expected_prob.iter()) {
    let e = n_f * p;
    if e < 1.0 {
      return Err(StatsError::DegenerateData(
        "a bin's expected count is below 1",
      ));
    }
    let d = o as f64 - e;
    statistic += d * d / e;
  }

  let df = observed.len() - 1;
  use stochastic_rs_distributions::DistributionExt;
  use stochastic_rs_distributions::chi_square::SimdChiSquared;
  let chi2 = SimdChiSquared::<f64>::new(df as f64);
  let cdf = chi2
    .cdf(statistic)
    .expect("SimdChiSquared has a closed-form cdf");
  let p_value = (1.0 - cdf).clamp(0.0, 1.0);
  let critical_value = chi2
    .quantile(1.0 - cfg.alpha)
    .expect("SimdChiSquared has a closed-form quantile");

  Ok(ChiSquareGofTest {
    statistic,
    p_value: Some(p_value),
    critical_values: Some(CriticalValues::new([(cfg.alpha, critical_value)])),
    rejects_null: p_value < cfg.alpha,
    df,
  })
}

/// Pools `cdf`'s masses over `[k_lo, k_hi]` and both open tails into bins `[edges[i], edges[i+1])`,
/// each of expected count `>= min_expected`; panics unless `k_lo <= k_hi` and `min_expected > 0`.
pub fn pool_integer_bins(
  n: u64,
  k_lo: i64,
  k_hi: i64,
  cdf: impl Fn(i64) -> f64,
  min_expected: f64,
) -> (Vec<i64>, Vec<f64>) {
  assert!(k_hi >= k_lo, "k_hi must be >= k_lo");
  assert!(min_expected > 0.0, "min_expected must be positive");

  let n_f = n as f64;
  let pmf = |k: i64| (cdf(k) - cdf(k - 1)).max(0.0);
  let left_tail = cdf(k_lo - 1).clamp(0.0, 1.0);
  let right_tail = (1.0 - cdf(k_hi)).clamp(0.0, 1.0);

  let mut edges = vec![i64::MIN];
  let mut probs = Vec::new();
  let mut acc = left_tail;
  for k in k_lo..=k_hi {
    acc += pmf(k);
    if n_f * acc >= min_expected {
      edges.push(k + 1);
      probs.push(acc);
      acc = 0.0;
    }
  }
  acc += right_tail;
  if probs.is_empty() || n_f * acc >= min_expected {
    edges.push(i64::MAX);
    probs.push(acc);
  } else {
    // Leftover tail mass too small to stand as its own bin: fold into the
    // last closed bin rather than leaving an under-powered final cell.
    *probs.last_mut().expect("probs checked non-empty above") += acc;
    *edges.last_mut().expect("edges checked non-empty above") = i64::MAX;
  }
  (edges, probs)
}

/// Counts `samples` into the bins of [`pool_integer_bins`]; panics unless `edges.len() >= 2`.
pub fn bin_observed(samples: &[i64], edges: &[i64]) -> Vec<u64> {
  assert!(edges.len() >= 2, "need at least one bin (2 edges)");
  let mut counts = vec![0u64; edges.len() - 1];
  let last = counts.len() - 1;
  for &x in samples {
    let idx = (edges.partition_point(|&e| e <= x) - 1).min(last);
    counts[idx] += 1;
  }
  counts
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::DistributionExt;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::poisson::SimdPoisson;

  use super::ChiSquareGofConfig;
  use super::bin_observed;
  use super::chi_square_critical_value;
  use super::chi_square_gof_test;
  use super::pool_integer_bins;
  use crate::StatsError;

  /// `chi2_{0.05, df}` for `df` in {1, 4, 9, 10}, as in the NIST/SEMATECH e-Handbook's table of
  /// chi-square percentage points.
  #[test]
  fn critical_value_matches_standard_table() {
    let table = [(1, 3.841), (4, 9.488), (9, 16.919), (10, 18.307)];
    for (df, expected) in table {
      let c = chi_square_critical_value(0.05, df);
      assert!(
        (c - expected).abs() < 1e-2,
        "chi2_(0.05,{df}) = {c}, expected {expected}"
      );
    }
  }

  /// `pool_integer_bins` must always partition the full mass exactly
  /// (telescoping cdf differences), regardless of window tightness.
  #[test]
  fn pool_integer_bins_conserves_total_mass() {
    let dist = SimdPoisson::<u64>::new(10.0);
    let (_, probs) = pool_integer_bins(20_000, 0, 30, |k| dist.cdf(k as f64).unwrap(), 5.0);
    let total = probs.iter().sum::<f64>();
    assert!((total - 1.0).abs() < 1e-9, "bin mass sums to {total}");
  }

  /// Every pooled bin must clear the requested `min_expected` floor.
  #[test]
  fn pool_integer_bins_respects_min_expected() {
    let dist = SimdPoisson::<u64>::new(10.0);
    let n = 5_000u64;
    let (_, probs) = pool_integer_bins(n, 0, 30, |k| dist.cdf(k as f64).unwrap(), 5.0);
    for &p in &probs {
      assert!(
        n as f64 * p >= 5.0 - 1e-9,
        "bin expected count {} below 5",
        n as f64 * p
      );
    }
  }

  /// A true Poisson(10) sample must not reject at alpha=0.05 — best of
  /// three pinned seeds, per this workspace's multi-seed mandate.
  #[test]
  fn accepts_true_poisson_sample() {
    const N: usize = 20_000;
    let best_p = [7u64, 123, 555]
      .into_iter()
      .map(|seed| {
        let dist = SimdPoisson::<u64>::new(10.0);
        let mut stream = dist.clone().seeded(&Deterministic::new(seed));
        let samples = (0..N).map(|_| stream.sample() as i64).collect::<Vec<_>>();
        let (edges, expected_prob) =
          pool_integer_bins(N as u64, 0, 35, |k| dist.cdf(k.max(0) as f64).unwrap(), 5.0);
        let observed = bin_observed(&samples, &edges);
        chi_square_gof_test(&observed, &expected_prob, ChiSquareGofConfig::default())
          .unwrap()
          .p_value
          .unwrap()
      })
      .fold(0.0_f64, f64::max);
    assert!(
      best_p > 0.01,
      "every seed gave p <= 0.01 (best {best_p}); likely a bug, not bad luck"
    );
  }

  #[test]
  fn one_bin_is_insufficient_data() {
    assert_eq!(
      chi_square_gof_test(&[10], &[1.0], ChiSquareGofConfig::default()).unwrap_err(),
      StatsError::InsufficientData { needed: 2, got: 1 }
    );
  }

  #[test]
  fn an_expected_count_below_one_is_degenerate_data() {
    assert_eq!(
      chi_square_gof_test(&[3, 1], &[0.9, 0.1], ChiSquareGofConfig::default()).unwrap_err(),
      StatsError::DegenerateData("a bin's expected count is below 1")
    );
  }

  #[test]
  #[should_panic(
    expected = "expected_prob must satisfy `expected_prob.len() == observed.len()`, got expected_prob.len() = 2, observed.len() = 3"
  )]
  fn unequal_lengths_are_a_precondition() {
    let _ = chi_square_gof_test(&[1, 2, 3], &[0.5, 0.5], ChiSquareGofConfig::default());
  }

  #[test]
  #[should_panic(
    expected = "expected_prob must satisfy `(expected_prob.iter().sum() - 1.0).abs() < 1e-6`, got expected_prob.iter().sum() = 0.9"
  )]
  fn masses_off_one_are_a_precondition() {
    let _ = chi_square_gof_test(&[10, 20], &[0.4, 0.5], ChiSquareGofConfig::default());
  }

  /// Poisson(10) samples tested against Poisson(16)'s bins must reject.
  #[test]
  fn rejects_mismatched_poisson_rate() {
    const N: usize = 20_000;
    let mut sampler = SimdPoisson::<u64>::new(10.0).seeded(&Deterministic::new(3));
    let reference = SimdPoisson::<u64>::new(16.0);
    let samples = (0..N).map(|_| sampler.sample() as i64).collect::<Vec<_>>();
    let (edges, expected_prob) = pool_integer_bins(
      N as u64,
      0,
      45,
      |k| reference.cdf(k.max(0) as f64).unwrap(),
      5.0,
    );
    let observed = bin_observed(&samples, &edges);
    let res =
      chi_square_gof_test(&observed, &expected_prob, ChiSquareGofConfig::default()).unwrap();
    assert!(res.rejects_null, "expected rejection, got {res:?}");
  }
}
