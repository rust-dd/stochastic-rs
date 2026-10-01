//! # Monte Carlo Methods
//!
//! $$
//! \hat{\mu}_N = \frac{1}{N}\sum_{i=1}^{N} f(X_i),\quad
//! \operatorname{Var}[\hat{\mu}_N] = \frac{\sigma^2}{N}
//! $$
//!
//! Variance reduction, quasi-Monte Carlo sequences (Sobol on the full
//! Joe-Kuo table with Owen-type scrambling, Halton) and their Brownian-bridge
//! path construction, multi-level MC, and American option pricing via
//! Longstaff-Schwartz.
//!
//! Reference: Glasserman (2003), *Monte Carlo Methods in Financial Engineering*,
//! DOI: 10.1007/978-0-387-21617-1

pub mod antithetic;
pub mod brownian_bridge_qmc;
pub mod control_variates;
pub mod halton;
pub mod importance_sampling;

pub mod lsm;
pub mod mlmc;
pub mod sobol;
pub mod stratified;

use ndarray::parallel::prelude::*;

use crate::traits::FloatExt;
use crate::traits::process::chunk_count;
use crate::traits::process::chunk_lens;

/// Result of a Monte Carlo estimation.
#[derive(Debug, Clone)]
pub struct McEstimate<T: FloatExt> {
  /// Estimated mean.
  pub mean: T,
  /// Standard error of the estimate: the sample standard deviation (`n − 1`
  /// denominator) over `√n`. `NaN` when fewer than two samples were used.
  pub std_err: T,
  /// Number of samples used.
  pub n_samples: usize,
}

impl<T: FloatExt> McEstimate<T> {
  /// Symmetric confidence interval `[mean ± z · std_err]`.
  pub fn confidence_interval(&self, z: T) -> (T, T) {
    (self.mean - z * self.std_err, self.mean + z * self.std_err)
  }

  /// 95% confidence interval (z = 1.96).
  pub fn ci_95(&self) -> (T, T) {
    self.confidence_interval(T::from_f64_fast(1.96))
  }
}

impl<T: FloatExt + std::fmt::Display> std::fmt::Display for McEstimate<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{:.4} ± {:.4}", self.mean, self.std_err)
  }
}

/// Streaming mean and variance in `f64`: Welford's update per sample, Chan–Golub–LeVeque's
/// pairwise update on `merge`. Immune to the `sum_sq / n − mean²` cancellation and the `f32` stall.
///
/// ```
/// use stochastic_rs_stochastic::mc::Welford;
///
/// let mut left = Welford::default();
/// left.extend([1.0_f64, 2.0, 3.0]);
/// let mut right = Welford::default();
/// right.extend([10.0_f64, 20.0]);
/// left.merge(&right);
///
/// assert_eq!(left.count(), 5);
/// assert!((left.mean() - 7.2).abs() < 1e-12);
/// assert!((left.sample_variance() - 63.7).abs() < 1e-12);
/// ```
///
/// Welford (1962), "Note on a Method for Calculating Corrected Sums of Squares and Products", DOI: 10.1080/00401706.1962.10490022.
/// Chan, Golub, LeVeque (1983), "Algorithms for Computing the Sample Variance: Analysis and Recommendations", DOI: 10.1080/00031305.1983.10483115.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Welford {
  n: usize,
  mean: f64,
  m2: f64,
}

impl Welford {
  /// Folds one sample in with Welford's update. The sample is converted to
  /// `f64`.
  pub fn push<T: FloatExt>(&mut self, y: T) {
    let y = y.to_f64().unwrap_or(f64::NAN);
    self.n += 1;
    let delta = y - self.mean;
    self.mean += delta / self.n as f64;
    self.m2 += delta * (y - self.mean);
  }

  /// Combines `other` into `self` as if its samples followed these (Chan, Golub and LeVeque 1983,
  /// eq. 1.5b). The bits depend on the merge order, so a reproducible reduction fixes it.
  pub fn merge(&mut self, other: &Welford) {
    if other.n == 0 {
      return;
    }
    if self.n == 0 {
      *self = other.clone();
      return;
    }
    let (m, n) = (self.n as f64, other.n as f64);
    let delta = other.mean - self.mean;
    self.mean += delta * (n / (m + n));
    self.m2 += other.m2 + delta * delta * (m * n / (m + n));
    self.n += other.n;
  }

  /// Number of samples folded in so far.
  pub fn count(&self) -> usize {
    self.n
  }

  /// Mean of the samples, `NaN` before the first one.
  pub fn mean(&self) -> f64 {
    if self.n == 0 { f64::NAN } else { self.mean }
  }

  /// Sample variance on the `n − 1` denominator, floored at zero. `NaN` for
  /// fewer than two samples, and once a `NaN` or infinite sample has been seen.
  pub fn sample_variance(&self) -> f64 {
    if self.n < 2 {
      return f64::NAN;
    }
    // `clamp`, not `max`: `NaN.max(0.0)` is `0.0`, which would report a
    // poisoned stream as exactly known.
    self.m2.clamp(0.0, f64::INFINITY) / (self.n as f64 - 1.0)
  }

  /// Standard error of the mean, `sqrt(s² / n)` for the sample variance `s²`.
  /// `NaN` wherever [`sample_variance`](Self::sample_variance) is.
  pub fn std_err(&self) -> f64 {
    (self.sample_variance() / self.n as f64).sqrt()
  }
}

/// Folds each sample in with [`push`](Welford::push).
impl<T: FloatExt> Extend<T> for Welford {
  fn extend<I: IntoIterator<Item = T>>(&mut self, samples: I) {
    for y in samples {
      self.push(y);
    }
  }
}

/// The estimate the accumulated samples give: their mean, its standard error
/// and their count.
impl<T: FloatExt> From<&Welford> for McEstimate<T> {
  fn from(acc: &Welford) -> Self {
    McEstimate {
      mean: T::from_f64_fast(acc.mean()),
      std_err: T::from_f64_fast(acc.std_err()),
      n_samples: acc.count(),
    }
  }
}

/// Mean and standard error of i.i.d. samples through [`Welford`]: `sqrt(s² / n)` on the `n − 1`
/// sample variance, `NaN` with fewer than two samples or a non-finite one.
///
/// Welford (1962), "Note on a Method for Calculating Corrected Sums of Squares and Products", DOI: 10.1080/00401706.1962.10490022.
pub(crate) fn estimate_from_samples<T: FloatExt>(
  samples: impl IntoIterator<Item = T>,
) -> McEstimate<T> {
  let mut acc = Welford::default();
  acc.extend(samples);
  McEstimate::from(&acc)
}

/// Run lengths of a parallel reduction over `n` samples: the split `sample_par` uses, a function of
/// `n` alone and never of the pool, because the boundaries fix the merged result's bits.
fn par_chunk_lens(n: usize) -> impl Iterator<Item = usize> {
  chunk_lens(n, chunk_count(n))
}

/// Folds `sample(0..n)` into one [`Welford`] without collecting: runs fold in parallel and merge in
/// run order, so the bits never depend on the pool size, as they would under rayon's `reduce`.
pub(crate) fn par_welford<T: FloatExt>(n: usize, sample: impl Fn(usize) -> T + Sync) -> Welford {
  let mut start = 0;
  let runs = par_chunk_lens(n)
    .map(|len| {
      let run = start..start + len;
      start += len;
      run
    })
    .collect::<Vec<_>>();
  let parts = runs
    .into_par_iter()
    .map(|run| {
      let mut acc = Welford::default();
      acc.extend(run.map(&sample));
      acc
    })
    .collect::<Vec<_>>();
  let mut total = Welford::default();
  for part in &parts {
    total.merge(part);
  }
  total
}

#[cfg(test)]
mod estimator_numerics {
  use ndarray::Array1;
  use rayon::ThreadPoolBuilder;

  use super::*;

  const PATHS: usize = 10_000;

  /// A constant with no exact binary form, so its sums round. `3.0` is exact in
  /// binary, which hides a variance computed as `sum_sq / n − mean²`.
  const NEAR_CONSTANT: f64 = 0.1;

  fn assert_zero_error(est: McEstimate<f64>) {
    assert!(
      (est.mean - NEAR_CONSTANT).abs() < 1e-12,
      "mean {}",
      est.mean
    );
    assert_eq!(est.std_err, 0.0);
    assert_eq!(est.n_samples, PATHS);
  }

  #[test]
  fn a_constant_payoff_has_zero_standard_error() {
    let est = antithetic::estimate::<f64, _>(10_000, 1, |_| 3.0);
    assert_eq!(est.mean, 3.0);
    assert_eq!(est.std_err, 0.0);
  }

  #[test]
  fn antithetic_reports_zero_error_for_a_near_constant_payoff() {
    assert_zero_error(antithetic::estimate(PATHS, 1, |_| NEAR_CONSTANT));
  }

  #[test]
  fn antithetic_par_reports_zero_error_for_a_near_constant_payoff() {
    assert_zero_error(antithetic::estimate_par(PATHS, 1, |_| NEAR_CONSTANT));
  }

  #[test]
  fn importance_sampling_reports_zero_error_for_a_near_constant_payoff() {
    let shift = Array1::<f64>::zeros(1);
    assert_zero_error(importance_sampling::estimate(
      PATHS,
      1,
      |_| NEAR_CONSTANT,
      &shift,
    ));
  }

  #[test]
  fn stratified_reports_zero_error_for_a_near_constant_payoff() {
    assert_zero_error(stratified::estimate(PATHS, 1, |_| NEAR_CONSTANT));
  }

  #[test]
  fn control_variates_report_zero_error_for_a_near_constant_payoff() {
    assert_zero_error(control_variates::estimate(
      PATHS,
      1,
      |_| NEAR_CONSTANT,
      |z: &Array1<f64>| z[0],
      0.0,
    ));
  }

  #[test]
  fn f32_means_do_not_saturate() {
    let est = estimate_from_samples(std::iter::repeat_n(1.0f32, 20_000_000));
    assert!((est.mean - 1.0).abs() < 1e-6, "mean {}", est.mean);
    assert_eq!(est.n_samples, 20_000_000);
  }

  #[test]
  fn the_helper_uses_the_sample_variance() {
    let est = estimate_from_samples([1.0f64, 2.0, 3.0, 4.0]);
    assert!((est.mean - 2.5).abs() < 1e-15);
    let var = 5.0 / 3.0;
    assert!((est.std_err - (var / 4.0f64).sqrt()).abs() < 1e-15);
  }

  #[test]
  fn a_large_offset_does_not_cancel() {
    let est = estimate_from_samples([1.0e9 + 1.0, 1.0e9 + 2.0, 1.0e9 + 3.0, 1.0e9 + 4.0]);
    assert_eq!(est.mean, 1.0e9 + 2.5);
    assert!((est.std_err - (5.0 / 3.0 / 4.0f64).sqrt()).abs() < 1e-12);
  }

  #[test]
  fn a_non_finite_sample_poisons_the_standard_error() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
      let est = estimate_from_samples([1.0, bad, 3.0]);
      assert!(
        est.std_err.is_nan(),
        "sample {bad} gave std_err {}",
        est.std_err
      );
    }
  }

  #[test]
  fn fewer_than_two_samples_have_no_standard_error() {
    let none = estimate_from_samples(std::iter::empty::<f64>());
    assert!(none.mean.is_nan() && none.std_err.is_nan());
    assert_eq!(none.n_samples, 0);

    let one = estimate_from_samples([2.5f64]);
    assert_eq!(one.mean, 2.5);
    assert!(one.std_err.is_nan());
    assert_eq!(one.n_samples, 1);
  }

  /// A pure function of the sample index, on an offset with a small spread.
  fn indexed_sample(i: usize) -> f64 {
    1_000.0 + (i as f64).sin()
  }

  fn sequential(samples: impl IntoIterator<Item = f64>) -> Welford {
    let mut acc = Welford::default();
    acc.extend(samples);
    acc
  }

  /// The chunked reduction written out sequentially: one accumulator per
  /// chunk, merged in chunk order.
  fn chunked_in_order(n: usize) -> Welford {
    let mut total = Welford::default();
    let mut start = 0;
    for len in par_chunk_lens(n) {
      total.merge(&sequential((start..start + len).map(indexed_sample)));
      start += len;
    }
    total
  }

  #[test]
  fn merging_chunks_matches_one_stream() {
    let n = 10_007;
    let whole = sequential((0..n).map(indexed_sample));
    // Uneven boundaries, with an empty chunk and a one-sample chunk.
    let bounds = [0, 1, 1, 17, 4_000, 4_001, n];
    let mut merged = Welford::default();
    for pair in bounds.windows(2) {
      merged.merge(&sequential((pair[0]..pair[1]).map(indexed_sample)));
    }
    assert_eq!(merged.count(), whole.count());
    assert!(
      (merged.mean() - whole.mean()).abs() < 1e-9,
      "mean {} against {}",
      merged.mean(),
      whole.mean()
    );
    let rel = (merged.sample_variance() - whole.sample_variance()) / whole.sample_variance();
    assert!(rel.abs() < 1e-10, "variance relative error {rel}");
  }

  /// Chan, Golub and LeVeque (1983), eq. (1.5b): `S = S_a + S_b + m n / (m + n) ·
  /// (mean_b − mean_a)²`. For `[1, 2, 3]` and `[10, 20]`: `2 + 50 + (3·2/5)·13²`.
  #[test]
  fn merge_is_the_pairwise_update_of_chan_golub_and_leveque() {
    let mut acc = sequential([1.0, 2.0, 3.0]);
    acc.merge(&sequential([10.0, 20.0]));
    assert_eq!(acc.count(), 5);
    assert!((acc.mean() - 7.2).abs() < 1e-12);
    assert!((acc.sample_variance() - 254.8 / 4.0).abs() < 1e-12);
  }

  #[test]
  fn merging_an_empty_accumulator_changes_nothing() {
    let full = sequential([1.5, 2.5, 4.0]);

    let mut into_empty = Welford::default();
    into_empty.merge(&full);
    assert_eq!(into_empty, full);

    let mut from_empty = full.clone();
    from_empty.merge(&Welford::default());
    assert_eq!(from_empty, full);
  }

  #[test]
  fn a_poisoned_chunk_poisons_the_merge() {
    let mut acc = sequential([1.0, 2.0]);
    acc.merge(&sequential([3.0, f64::NAN]));
    assert!(acc.sample_variance().is_nan());
    assert!(acc.std_err().is_nan());
  }

  /// The chunking is a function of `n` alone, below and above the cap, so the
  /// reduction is bit-identical whatever the rayon pool looks like.
  #[test]
  fn the_parallel_reduction_ignores_the_pool_size() {
    for n in [0, 1, 7, 64, 65, 1_000, 12_345] {
      let reference = chunked_in_order(n);
      for threads in [1, 4] {
        let pool = ThreadPoolBuilder::new()
          .num_threads(threads)
          .build()
          .unwrap();
        let acc = pool.install(|| par_welford(n, indexed_sample));
        assert_eq!(acc, reference, "n = {n}, {threads} threads");
      }
      let whole = sequential((0..n).map(indexed_sample));
      assert_eq!(reference.count(), whole.count());
      if n > 1 {
        let rel = (reference.sample_variance() - whole.sample_variance()) / whole.sample_variance();
        assert!(rel.abs() < 1e-10, "n = {n}: variance relative error {rel}");
      }
    }
  }
}
