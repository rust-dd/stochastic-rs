//! Streaming mean and variance, and the parallel fold whose bits never depend on the pool size.

use ndarray::parallel::prelude::*;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
use crate::traits::process::chunk_count;
use crate::traits::process::chunk_lens;

/// Streaming mean and variance in `f64`: Welford's update per sample, Chan–Golub–LeVeque's
/// pairwise update on `merge`. Avoids the `sum_sq / n − mean²` cancellation and `f32` swamping.
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

/// Run lengths of a parallel reduction over `n` samples: the split `sample_par` uses, a function of
/// `n` alone and never of the pool, because the boundaries fix the merged result's bits.
fn par_chunk_lens(n: usize) -> impl Iterator<Item = usize> {
  chunk_lens(n, chunk_count(n))
}

/// Folds `sample(i, stream)` over `0..n` in parallel runs, each on a normal stream forked from
/// `seed` on this thread in run order and merged in run order: the bits never see the pool size.
pub(crate) fn par_welford_seeded<T: FloatExt, S: SeedExt>(
  n: usize,
  seed: &S,
  sample: impl Fn(usize, &mut Seeded<SimdNormal<T>>) -> T + Sync,
) -> Welford {
  let mut start = 0;
  let runs = par_chunk_lens(n)
    .map(|len| {
      let run = start..start + len;
      start += len;
      run
    })
    .collect::<Vec<_>>();
  let mut base = SimdNormal::<T>::new(T::zero(), T::one()).seeded(seed);
  let streams = (0..runs.len())
    .map(|run| base.fork(run as u64))
    .collect::<Vec<_>>();
  let parts = runs
    .into_par_iter()
    .zip(streams)
    .map(|(run, mut stream)| {
      let mut acc = Welford::default();
      acc.extend(run.map(|i| sample(i, &mut stream)));
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
mod tests {
  use rayon::ThreadPoolBuilder;
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

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
        let acc = pool.install(|| {
          par_welford_seeded::<f64, _>(n, &Deterministic::new(1), |i, _| indexed_sample(i))
        });
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
