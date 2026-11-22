//! Shared standard-error arithmetic for the bespoke Monte Carlo pricers.

use rayon::prelude::*;
use stochastic_rs_stochastic::mc::McEstimate;

/// Paths per chunk, a constant rather than the thread count, so the bits ignore the pool.
const CHUNK: usize = 256;

/// The estimate of `n` payoffs: chunks summed in parallel and folded in order, so one seed gives
/// one value on any pool; `scale` (a discount, a notional) multiplies both moments.
pub(crate) fn par_estimate(
  n: usize,
  scale: f64,
  payoff: impl Fn(usize) -> f64 + Sync + Send,
) -> McEstimate<f64> {
  // −0.0 is the identity of `+`, so a single chunk sums as `Iterator::sum` does.
  let add = |(sum, sum_sq): (f64, f64), (s, q): (f64, f64)| (sum + s, sum_sq + q);
  let moments = (0..n.div_ceil(CHUNK))
    .into_par_iter()
    .map(|chunk| {
      (chunk * CHUNK..n.min((chunk + 1) * CHUNK))
        .map(&payoff)
        .fold((-0.0, -0.0), |acc, y| add(acc, (y, y * y)))
    })
    .collect::<Vec<(f64, f64)>>();
  let (sum, sum_sq) = moments.into_iter().fold((-0.0, -0.0), add);
  McEstimate {
    mean: scale * sum / n as f64,
    std_err: scale.abs() * std_err_from_sums(sum, sum_sq, n),
    n_samples: n,
  }
}

/// Whether a Monte Carlo query has no price: a NaN coordinate, `tau < 0`, or one of `positive` (the
/// spots, and a strike the payoff needs positive) that is not positive before expiry.
pub(crate) fn off_domain(positive: &[f64], others: &[f64], tau: f64) -> bool {
  let nan = positive.iter().chain(others).any(|x| x.is_nan()) || tau.is_nan();
  nan || tau < 0.0 || (tau > 0.0 && positive.iter().any(|&x| x <= 0.0))
}

/// The estimate of a query no path is simulated for: exact for a number, NaN for a NaN.
pub(crate) fn pathless(mean: f64) -> McEstimate<f64> {
  McEstimate {
    mean,
    std_err: 0.0 * mean,
    n_samples: 0,
  }
}

/// Undiscounted `sqrt(((sum_sq − sum²/n)/(n − 1))/n)` (Glasserman 2003, §1.1): NaN for a NaN sum or
/// `n < 2`, a cancellation's tiny negative variance floored at zero.
pub(crate) fn std_err_from_sums(sum: f64, sum_sq: f64, n: usize) -> f64 {
  if n < 2 {
    return f64::NAN;
  }
  let nf = n as f64;
  let var = (sum_sq - sum * sum / nf) / (nf - 1.0);
  if var.is_nan() {
    return f64::NAN;
  }
  (var.max(0.0) / nf).sqrt()
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Cross-checked against the two-pass definition on the same data.
  #[test]
  fn matches_the_two_pass_definition() {
    let xs = [1.0_f64, 2.0, 4.0, 8.0, 16.0];
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let want = (var / n).sqrt();
    let sum = xs.iter().sum::<f64>();
    let sum_sq = xs.iter().map(|x| x * x).sum::<f64>();
    let got = std_err_from_sums(sum, sum_sq, xs.len());
    assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
  }

  #[test]
  fn a_poisoned_sum_stays_nan() {
    assert!(std_err_from_sums(f64::NAN, 1.0, 10).is_nan());
    assert!(std_err_from_sums(1.0, f64::NAN, 10).is_nan());
  }

  #[test]
  fn fewer_than_two_samples_have_no_estimate() {
    assert!(std_err_from_sums(1.0, 1.0, 1).is_nan());
  }

  /// Identical samples cancel to a tiny negative variance in floating
  /// point; the floor keeps the error at exactly zero.
  #[test]
  fn constant_payoffs_report_zero_error() {
    let sum = 0.1_f64 * 3.0;
    let sum_sq = 0.1_f64 * 0.1 * 3.0;
    assert_eq!(std_err_from_sums(sum, sum_sq, 3), 0.0);
  }

  #[test]
  fn the_chunks_fold_in_order_on_every_pool() {
    let n = 5 * CHUNK + 17;
    let payoff = |i: usize| (i as f64 * 0.618_033_988_749_895).fract() - 0.25;
    let (mut sum, mut sum_sq) = (-0.0, -0.0);
    for start in (0..n).step_by(CHUNK) {
      let chunk = (start..n.min(start + CHUNK)).map(payoff);
      let (s, q) = chunk.fold((-0.0, -0.0), |(s, q), y| (s + y, q + y * y));
      (sum, sum_sq) = (sum + s, sum_sq + q);
    }
    let want = (
      2.0 * sum / n as f64,
      2.0 * std_err_from_sums(sum, sum_sq, n),
    );
    let bits = |e: McEstimate<f64>| (e.mean.to_bits(), e.std_err.to_bits());
    for threads in [1, 4] {
      let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
      let got = pool.install(|| par_estimate(n, 2.0, payoff));
      assert_eq!(
        bits(got),
        (want.0.to_bits(), want.1.to_bits()),
        "{threads} threads"
      );
    }
    let got = par_estimate(n, 2.0, payoff);
    assert_eq!(
      bits(got),
      (want.0.to_bits(), want.1.to_bits()),
      "the global pool"
    );
  }
}
