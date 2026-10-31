use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::tests::scalar_draws;
use crate::tests::scalar_ks_best_p;

#[test]
fn truncated_normal_samples_within_bounds() {
  for seed in [2718, 999, 42] {
    let law = SimdTruncatedNormal::<f64>::new(0.0, 1.0, -1.0, 2.0);
    let mut tn = law.seeded(&Deterministic::new(seed));
    for _ in 0..5_000 {
      let x = tn.sample();
      assert!(
        (-1.0..=2.0).contains(&x),
        "seed {seed}: sample {x} out of [-1, 2]"
      );
    }
  }
}

/// Draws on every proposal leave the analytics unbuilt; equality and `Debug` read the parameters, not the cache.
#[test]
fn construction_and_draws_skip_the_quadrature() {
  for (lo, hi) in [(-1.0, 2.0), (3.0, 6.0), (1.5, 2.1)] {
    let law = SimdTruncatedNormal::<f64>::new(0.0, 1.0, lo, hi);
    let (twin, text) = (law.clone(), format!("{law:?}"));
    let draws = scalar_draws(&law, 7, 1_000);
    assert!(draws.iter().all(|x| (lo..=hi).contains(x)) && law.analytics.get().is_none());
    law.mean();
    assert!(law.analytics.get().is_some() && law == twin && format!("{law:?}") == text);
  }
}

/// The density, summed by midpoints with no reference to the engine's own mass, integrates to 1.
#[test]
fn truncated_normal_pdf_normalised() {
  let tn = SimdTruncatedNormal::<f64>::new(0.0, 1.0, -1.0, 2.0);
  let n = 1_000;
  let h = 3.0 / n as f64;
  let s: f64 = (0..n)
    .map(|k| tn.pdf(-1.0 + (k as f64 + 0.5) * h).unwrap() * h)
    .sum();
  assert!(
    (s - 1.0).abs() < 5e-3,
    "truncated normal pdf integrates to {s}, expected 1"
  );
}

/// Draws in bounds and a conditional mean against Simpson quadrature of `exp(-a t - t²/2)`, the density of `Z - a`,
/// which forms no cdf: the inverse transform put 17 % of [8, 8.5] and all of [10, 12] at `+inf`.
fn assert_far_tail_means(mut draws: impl FnMut(SimdTruncatedNormal<f64>) -> Vec<f64>) {
  for (lo, hi) in [(8.0_f64, 8.5_f64), (10.0, 12.0), (-12.0, -10.0)] {
    let xs = draws(SimdTruncatedNormal::<f64>::new(0.0, 1.0, lo, hi));
    for x in &xs {
      assert!((lo..=hi).contains(x), "sample {x} out of [{lo}, {hi}]");
    }
    let a = lo.abs().min(hi.abs());
    let width = hi - lo;
    let (mut num, mut den) = (0.0, 0.0);
    let steps = 4_000;
    let h = width / steps as f64;
    for k in 0..=steps {
      let t = k as f64 * h;
      let w = if k == 0 || k == steps {
        1.0
      } else if k % 2 == 1 {
        4.0
      } else {
        2.0
      };
      let f = (-a * t - 0.5 * t * t).exp();
      num += w * t * f;
      den += w * f;
    }
    let want = (a + num / den) * lo.signum();
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    assert!(
      (mean - want).abs() < 0.01,
      "[{lo}, {hi}]: mean = {mean}, expected ≈ {want}"
    );
  }
}

/// A far-tail interval keeps its draws inside its bounds and keeps the right conditional mean.
#[test]
fn truncated_normal_far_tail_stays_in_bounds() {
  assert_far_tail_means(|d| {
    let mut s = d.seeded(&Deterministic::new(2718));
    (0..40_000).map(|_| s.sample()).collect()
  });
}

/// The honest draw keeps the same far-tail bounds and conditional means on the caller's rng.
#[test]
fn scalar_far_tail_stays_in_bounds() {
  assert_far_tail_means(|d| scalar_draws(&d, 2718, 40_000));
}

/// The honest `Distribution` agrees with the cdf on the rejection, both tail proposals, the mirror and the inverse cdf.
#[test]
fn scalar_sample_matches_cdf() {
  for (lo, hi) in [
    (-1.0, 2.0),
    (3.0, 6.0),
    (3.0, 3.4),
    (3.0, 3.2),
    (1.5, 2.1),
    (0.0, 1.7),
    (8.0, 8.2),
    (-6.0, -3.0),
    (-0.05, 0.05),
  ] {
    let d = SimdTruncatedNormal::<f64>::new(0.0, 1.0, lo, hi);
    let best = scalar_ks_best_p(&d, |x| d.cdf(x).unwrap());
    assert!(best > 0.01, "[{lo}, {hi}]: best p = {best}");
  }
}

/// Each pair brackets Robert's (2.1) width at a = 3, 0 and 8 (0.317, 1.649, 0.124), inside the band where a `2/α*`
/// switch would take the uniform proposal.
#[test]
fn tail_proposal_follows_roberts_rule() {
  let exponential = |a: f64, b: f64| {
    matches!(
      TailSetup::new(a, b, false).proposal,
      TailProposal::Exponential { .. }
    )
  };
  assert!(exponential(3.0, 3.4));
  assert!(!exponential(3.0, 3.2));
  assert!(exponential(0.0, 1.7));
  assert!(!exponential(0.0, 1.6));
  assert!(exponential(8.0, 8.2));
  assert!(!exponential(8.0, 8.1));
  assert!(exponential(3.0, f64::INFINITY));
}

/// Outside the bounds the law answers from its support alone, before any analytic is built.
#[test]
fn truncated_pdf_cdf_outside_bounds() {
  let tn = SimdTruncatedNormal::<f64>::new(0.0, 1.0, -1.0, 1.0);
  assert_eq!(tn.pdf(-1.5).unwrap(), 0.0);
  assert_eq!(tn.pdf(1.5).unwrap(), 0.0);
  assert_eq!(tn.cdf(-2.0).unwrap(), 0.0);
  assert_eq!(tn.cdf(2.0).unwrap(), 1.0);
}

/// `mu()` and `sigma()` read back the parent's arguments, not moments of the truncated law.
#[test]
fn getters_return_the_parent_parameters() {
  let tn = SimdTruncatedNormal::<f64>::new(0.5, 2.0, 1.0, 4.0);
  assert_eq!(
    (tn.mu(), tn.sigma(), tn.lower(), tn.upper()),
    (0.5, 2.0, 1.0, 4.0)
  );
}
