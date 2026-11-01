use stochastic_rs_core::simd_rng::Deterministic;

use super::sampler::TailProposal;
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
      TailSetup::new(a, b, f64::NAN, false).proposal,
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

/// Past `MAX/2` the tail's rate no longer overflows: every draw returns, here the bound, where each law is a point to
/// its last double.
#[test]
fn draws_past_half_max_return() {
  let (max, inf) = (f64::MAX, f64::INFINITY);
  for (mu, sigma, lo, up, at) in [
    (0.0, 1.0, 1e308, max, 1e308),
    (0.0, 1.0, 9e307, inf, 9e307),
    (0.0, 1e-10, 1e298, inf, 1e298),
    (-max, 1.0, 1.0, 2.0, 1.0),
    (-max / 2.0, 1.0, max / 2.0, max, max / 2.0),
    (max, 1.0, -max, -1.0, -1.0),
  ] {
    let law = SimdTruncatedNormal::<f64>::new(mu, sigma, lo, up);
    let mut stream = law.clone().seeded(&Deterministic::new(7));
    let streamed = (0..100).map(|_| stream.sample());
    for x in scalar_draws(&law, 7, 100).into_iter().chain(streamed) {
      assert_eq!(x, at, "TN({mu:e}, {sigma:e}, [{lo:e}, {up:e}])");
    }
  }
}

/// Where the law spreads over the subnormals past `μ ± σ·MAX`, or its bounds round to one standardised double, the
/// draws follow the law the analytics answer.
#[test]
fn draws_past_mu_plus_sigma_max_follow_the_analytics() {
  let (max, inf) = (f64::MAX, f64::INFINITY);
  for (mu, sigma, lo, up) in [
    (-max, 0.9, 1e-310, 1.0),
    (max, 0.9, -1.0, -1e-310),
    (-max, 1.5, 0.0, inf),
    (-max, 1.0, 1e-310, 1.0),
  ] {
    let law = SimdTruncatedNormal::<f64>::new(mu, sigma, lo, up);
    let best = scalar_ks_best_p(&law, |x| law.cdf(x).unwrap());
    assert!(
      best > 0.01,
      "TN({mu:e}, {sigma}, [{lo:e}, {up:e}]): best p = {best}"
    );
  }
}
