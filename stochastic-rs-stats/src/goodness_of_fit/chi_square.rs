//! # Pearson chi-square goodness-of-fit test (discrete / binned data)
//!
//! Tests whether integer-count data was drawn from a **fully specified**
//! probability mass function — every parameter known in advance, not
//! fitted from the data under test (see
//! [`crate::goodness_of_fit::kolmogorov_smirnov`]'s module doc for why
//! that distinction matters; the same reasoning applies here).
//!
//! $$ \chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}, \qquad E_i = n \cdot p_i $$
//!
//! Reference: Pearson, K. (1900), "On the criterion that a given system
//! of deviations from the probable in the case of a correlated system
//! of variables is such that it can be reasonably supposed to have
//! arisen from random sampling", *Philosophical Magazine* Series 5,
//! 50(302), 157-175.
//!
//! For a fully specified null, `\chi^2` is asymptotically
//! `\chi^2_{k-1}` — one fewer than the bin count `k`, because the
//! observed bin proportions are constrained to sum to 1 (Pearson 1900).
//! This is one lower than the general `df = k - 1 - s` used when `s`
//! parameters are additionally *estimated from the same data* — not the
//! case here, so no further subtraction applies.
//!
//! ## Why chi-square rather than Kolmogorov-Smirnov here
//!
//! KS's asymptotic null distribution assumes a continuous `F`, so that
//! `F_n(x) - F(x)` is driven only by sampling noise; on an integer
//! lattice `F` has jumps at every support point and the same statistic
//! is no longer calibrated the same way, which is exactly why a
//! *binned* statistic is used for discrete distributions instead
//! (Pearson 1900 predates Kolmogorov 1933 and was designed for
//! categorical/count data from the start).
//!
//! ## Bin pooling rule
//!
//! Cochran, W.G. (1954), "Some methods for strengthening the common
//! chi-square tests", *Biometrics* 10(4), 417-451 — the chi-square
//! approximation degrades once expected cell counts get too small;
//! Cochran's widely used rule of thumb is that every cell's expected
//! count should be at least 5 (more precisely: no cell below 1, and no
//! more than 20% of cells below 5). [`pool_integer_bins`] enforces
//! "every bin's expected count `\ge` `min_expected`" by accumulating
//! adjacent lattice cells until the threshold is met, folding in the two
//! open-ended tails `(-\infty, k_{lo})` / `(k_{hi}, \infty)` as the
//! outermost bins; [`chi_square_gof_test`] additionally hard-panics
//! below Cochran's stricter floor of expected count `< 1`, since below
//! that the approximation is not considered valid at all.

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

/// Result of a Pearson chi-square goodness-of-fit test.
#[derive(Debug, Clone, Copy)]
pub struct ChiSquareGofResult {
  /// The `\chi^2` statistic.
  pub statistic: f64,
  /// Degrees of freedom, `bins - 1` (Pearson 1900; see module doc).
  pub df: usize,
  /// Upper-tail `\chi^2_{df}` critical value at `alpha`.
  pub critical_value: f64,
  /// p-value: `P(\chi^2_{df} > statistic)`.
  pub p_value: f64,
  /// Whether the null is rejected at `alpha`.
  pub reject: bool,
}

impl HypothesisTest for ChiSquareGofResult {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.reject)
  }
}

/// Upper-tail chi-square critical value: the `c` such that `P(X > c) =
/// alpha` for `X ~ ChiSquared(df)`. Computed via
/// [`SimdChiSquared`](stochastic_rs_distributions::chi_square::SimdChiSquared)'s
/// `DistributionExt::inv_cdf`, itself cross-validated against `statrs`
/// in `stochastic-rs-distributions/tests/distribution_ext_vs_reference.rs`
/// (`chi_squared_matches_statrs`).
pub fn chi_square_critical_value(alpha: f64, df: usize) -> f64 {
  assert!(alpha > 0.0 && alpha < 1.0, "alpha must be in (0, 1)");
  assert!(
    df >= 1,
    "chi-square test needs at least 1 degree of freedom"
  );
  use stochastic_rs_core::simd_rng::Unseeded;
  use stochastic_rs_distributions::DistributionExt;
  use stochastic_rs_distributions::chi_square::SimdChiSquared;
  SimdChiSquared::<f64>::new(df as f64, &Unseeded).inv_cdf(1.0 - alpha)
}

/// Pearson's chi-square goodness-of-fit statistic for pre-binned counts.
///
/// `observed[i]` is the sample count landing in bin `i`; `expected_prob[i]`
/// is that bin's probability mass under the fully specified null (must
/// sum to 1 across bins within `1e-6`, and every bin's expected count
/// `n * expected_prob[i]` — `n` the total observed count — must be at
/// least 1; see the module doc's "Bin pooling rule").
///
/// # Panics
/// Panics if `observed.len() != expected_prob.len()`, if fewer than 2
/// bins are given, if `expected_prob` does not sum to ~1, if any bin's
/// expected count is below 1, or if `cfg.alpha` is not in `(0, 1)`.
pub fn chi_square_gof_test(
  observed: &[u64],
  expected_prob: &[f64],
  cfg: ChiSquareGofConfig,
) -> ChiSquareGofResult {
  assert_eq!(
    observed.len(),
    expected_prob.len(),
    "observed/expected_prob length mismatch"
  );
  assert!(observed.len() >= 2, "chi-square GoF needs at least 2 bins");
  assert!(
    cfg.alpha > 0.0 && cfg.alpha < 1.0,
    "alpha must be in (0, 1)"
  );

  let n = observed.iter().sum::<u64>();
  let n_f = n as f64;
  let prob_sum = expected_prob.iter().sum::<f64>();
  assert!(
    (prob_sum - 1.0).abs() < 1e-6,
    "expected_prob must sum to 1, got {prob_sum}"
  );

  let mut statistic = 0.0_f64;
  for (&o, &p) in observed.iter().zip(expected_prob.iter()) {
    let e = n_f * p;
    assert!(
      e >= 1.0,
      "bin expected count {e} < 1 violates Cochran (1954)'s validity floor; widen the bins"
    );
    let d = o as f64 - e;
    statistic += d * d / e;
  }

  let df = observed.len() - 1;
  use stochastic_rs_core::simd_rng::Unseeded;
  use stochastic_rs_distributions::DistributionExt;
  use stochastic_rs_distributions::chi_square::SimdChiSquared;
  let chi2 = SimdChiSquared::<f64>::new(df as f64, &Unseeded);
  let p_value = (1.0 - chi2.cdf(statistic)).clamp(0.0, 1.0);
  let critical_value = chi2.inv_cdf(1.0 - cfg.alpha);

  ChiSquareGofResult {
    statistic,
    df,
    critical_value,
    p_value,
    reject: p_value < cfg.alpha,
  }
}

/// Pools an integer-lattice pmf's cells over `[k_lo, k_hi]` into bins
/// whose expected count clears `min_expected` (Cochran 1954; see module
/// doc), folding the two open-ended tails `(-\infty, k_{lo})` /
/// `(k_{hi}, \infty)` in as the outermost bins so the returned bins
/// always partition the *entire* integer line — no probability mass is
/// dropped regardless of how tight `[k_lo, k_hi]` is chosen.
///
/// `cdf(k)` must return `P(K \le k)` for the candidate distribution.
/// Bin `i` covers `[edges[i], edges[i+1])`; `edges[0] = i64::MIN` and
/// `edges[last] = i64::MAX` are open-ended sentinels, so
/// `edges.len() == expected_prob.len() + 1`.
///
/// # Panics
/// Panics if `k_hi < k_lo` or `min_expected <= 0`.
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

/// Counts `samples` into the bins produced by [`pool_integer_bins`].
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
  use stochastic_rs_distributions::poisson::SimdPoisson;

  use super::ChiSquareGofConfig;
  use super::bin_observed;
  use super::chi_square_critical_value;
  use super::chi_square_gof_test;
  use super::pool_integer_bins;

  /// Standard textbook table: `chi2_{0.05, df}` for df in {1, 4, 9, 10}.
  /// Values from any standard chi-square table (e.g. NIST/SEMATECH
  /// e-Handbook of Statistical Methods, chi-square percentage points).
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
    let dist = SimdPoisson::<u64>::new(10.0, &Deterministic::new(1));
    let (_, probs) = pool_integer_bins(20_000, 0, 30, |k| dist.cdf(k as f64), 5.0);
    let total = probs.iter().sum::<f64>();
    assert!((total - 1.0).abs() < 1e-9, "bin mass sums to {total}");
  }

  /// Every pooled bin must clear the requested `min_expected` floor.
  #[test]
  fn pool_integer_bins_respects_min_expected() {
    let dist = SimdPoisson::<u64>::new(10.0, &Deterministic::new(1));
    let n = 5_000u64;
    let (_, probs) = pool_integer_bins(n, 0, 30, |k| dist.cdf(k as f64), 5.0);
    for &p in &probs {
      assert!(
        n as f64 * p >= 5.0 - 1e-9,
        "bin expected count {} below 5",
        n as f64 * p
      );
    }
  }

  /// A true Poisson(10) sample must not reject at alpha=0.05 — worst of
  /// three pinned seeds, per this workspace's multi-seed mandate.
  #[test]
  fn accepts_true_poisson_sample() {
    const N: usize = 20_000;
    let worst_p = [7u64, 123, 555]
      .into_iter()
      .map(|seed| {
        let dist = SimdPoisson::<u64>::new(10.0, &Deterministic::new(seed));
        let samples = (0..N)
          .map(|_| dist.sample_fast() as i64)
          .collect::<Vec<_>>();
        let (edges, expected_prob) =
          pool_integer_bins(N as u64, 0, 35, |k| dist.cdf(k.max(0) as f64), 5.0);
        let observed = bin_observed(&samples, &edges);
        chi_square_gof_test(&observed, &expected_prob, ChiSquareGofConfig::default()).p_value
      })
      .fold(1.0_f64, f64::min);
    assert!(
      worst_p > 0.01,
      "every seed gave p <= 0.01 (worst {worst_p}); likely a bug, not bad luck"
    );
  }

  /// Poisson(10) samples tested against Poisson(16)'s bins must reject.
  #[test]
  fn rejects_mismatched_poisson_rate() {
    const N: usize = 20_000;
    let sampler = SimdPoisson::<u64>::new(10.0, &Deterministic::new(3));
    let reference = SimdPoisson::<u64>::new(16.0, &Deterministic::new(3));
    let samples = (0..N)
      .map(|_| sampler.sample_fast() as i64)
      .collect::<Vec<_>>();
    let (edges, expected_prob) =
      pool_integer_bins(N as u64, 0, 45, |k| reference.cdf(k.max(0) as f64), 5.0);
    let observed = bin_observed(&samples, &edges);
    let res = chi_square_gof_test(&observed, &expected_prob, ChiSquareGofConfig::default());
    assert!(res.reject, "expected rejection, got {res:?}");
  }
}
