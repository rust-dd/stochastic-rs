use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::KolmogorovSmirnovConfig;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::kolmogorov_smirnov_test;

use super::SimdNormal;
use crate::traits::DistributionExt as _;

const SEEDS: [u64; 3] = [2718, 999, 42];

fn mean(samples: &[f64]) -> f64 {
  samples.iter().sum::<f64>() / samples.len() as f64
}

/// KS against the sampler's own `cdf` (Kolmogorov 1933 / Smirnov 1948 /
/// Massey 1951 critical values, alpha=0.05 — see
/// `stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov`'s module
/// doc), worst-of-three pinned seeds (see [`SEEDS`]): a correct test
/// still rejects a true null at rate alpha, and the SIMD stream differs
/// across platforms, so one seed cannot be trusted to be lucky
/// everywhere. Replaces this file's former `ks_critical = 2.0/sqrt(N)`
/// bound, which implied an undeclared alpha of roughly 0.0007.
fn worst_ks_p_value(
  n: usize,
  make_dist: impl Fn(u64) -> (Vec<f64>, Box<dyn Fn(f64) -> f64>),
) -> f64 {
  SEEDS
    .into_iter()
    .map(|seed| {
      let (samples, cdf) = make_dist(seed);
      assert_eq!(samples.len(), n);
      kolmogorov_smirnov_test(
        ArrayView1::from(&samples),
        cdf,
        KolmogorovSmirnovConfig::default(),
      )
      .p_value
    })
    .fold(1.0_f64, f64::min)
}

/// The dual-engine pair path interleaves batches from engines A and B —
/// every lane (including B's) must still be N(0, 1).
#[cfg(feature = "dual-stream-rng")]
#[test]
fn simd_normal_dual_pair_path_matches_theoretical_distribution() {
  const N: usize = 40_000;
  let worst_p = worst_ks_p_value(N, |seed| {
    let dist = crate::SimdNormalDual::<f64>::new(0.0, 1.0, &Deterministic::new(seed));
    let mut samples = vec![0.0_f64; N];
    dist.fill_standard_fast(&mut samples);
    assert!(samples.iter().all(|x| x.is_finite()));
    (samples, Box::new(move |x| dist.cdf(x)))
  });
  assert!(
    worst_p > 0.01,
    "every seed gave p <= 0.01 (worst {worst_p}); likely a bug, not bad luck"
  );
}

#[test]
fn simd_normal_matches_theoretical_distribution() {
  const N: usize = 40_000;
  let mu = -0.75_f64;
  let sigma = 1.35_f64;

  let dist = SimdNormal::<f64>::new(mu, sigma, &Deterministic::new(0x4e07));
  let mut samples = vec![0.0_f64; N];
  dist.fill_slice(&mut samples);

  assert!(
    samples.iter().all(|x| x.is_finite()),
    "non-finite normal sample encountered"
  );

  let mean_emp = mean(&samples);
  let mean_se = sigma / (N as f64).sqrt();
  assert!(
    (mean_emp - mu).abs() < 6.0 * mean_se,
    "normal mean mismatch: emp={mean_emp}, target={mu}, se={mean_se}"
  );

  let worst_p = worst_ks_p_value(N, |seed| {
    let dist = SimdNormal::<f64>::new(mu, sigma, &Deterministic::new(seed));
    let mut samples = vec![0.0_f64; N];
    dist.fill_slice(&mut samples);
    (samples, Box::new(move |x| dist.cdf(x)))
  });
  assert!(
    worst_p > 0.01,
    "every seed gave p <= 0.01 (worst {worst_p}); likely a bug, not bad luck"
  );
}

/// The ziggurat's tail, which a whole-distribution statistic cannot see.
///
/// Marsaglia & Tsang's algorithm draws `|z| > 3.442620` from a separate
/// exponential rejection step, and until September 2026 this crate negated
/// that draw's offset: every tail sample landed *inside* the boundary
/// instead of past it, so no `SimdNormal` ever returned a four-sigma move.
/// The KS test above passed throughout — it can only see 5.8e-4 of misplaced
/// mass as a 5.8e-4 shift in the CDF, a hundredth of its critical value at
/// these sample sizes — which is why the tail needs a tail statistic.
///
/// `P(|z| > 4) = 6.3342e-5` (Abramowitz & Stegun 26.2.19), so 12 million
/// draws hold about 760 of them; the band is five Poisson standard
/// deviations wide, and a folded tail gives exactly zero.
#[test]
fn the_ziggurat_tail_reaches_past_its_boundary() {
  const CHUNK: usize = 1_000_000;
  const CHUNKS: usize = 4;
  const TAIL_PROBABILITY: f64 = 6.334_248_366_623_984e-5;
  let (mut counted, mut widest) = (0usize, 0.0_f64);
  for seed in SEEDS {
    let dist = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(seed));
    let mut samples = vec![0.0_f64; CHUNK];
    for _ in 0..CHUNKS {
      dist.fill_slice(&mut samples);
      counted += samples.iter().filter(|z| z.abs() > 4.0).count();
      widest = samples.iter().fold(widest, |w, z| w.max(z.abs()));
    }
  }
  let drawn = (SEEDS.len() * CHUNKS * CHUNK) as f64;
  let expected = drawn * TAIL_PROBABILITY;
  let band = 5.0 * expected.sqrt();
  assert!(
    (counted as f64 - expected).abs() < band,
    "{counted} draws past four sigma in {drawn}, the law expects {expected:.0} ± {band:.0}"
  );
  assert!(
    widest > 4.5,
    "the widest of {drawn} draws was {widest}, so the tail is cut off"
  );
}

/// The moments the tail carries: a normal's kurtosis is 3 and its sixth
/// moment 15, and both are dominated by draws the ziggurat's rejection step
/// produces. With the tail folded they came out at 2.96 and 14.15 — the
/// shortfall that made an ARCH(1)'s sampled kurtosis miss Engle's (1982)
/// closed form by five standard errors.
#[test]
fn the_fourth_and_sixth_moments_match_the_normal() {
  const CHUNK: usize = 1_000_000;
  const CHUNKS: usize = 4;
  let (mut m2, mut m4, mut m6) = (0.0_f64, 0.0_f64, 0.0_f64);
  for seed in SEEDS {
    let dist = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(seed));
    let mut samples = vec![0.0_f64; CHUNK];
    for _ in 0..CHUNKS {
      dist.fill_slice(&mut samples);
      m2 += samples.iter().map(|z| z * z).sum::<f64>();
      m4 += samples.iter().map(|z| z.powi(4)).sum::<f64>();
      m6 += samples.iter().map(|z| z.powi(6)).sum::<f64>();
    }
  }
  let drawn = (SEEDS.len() * CHUNKS * CHUNK) as f64;
  let (m2, m4, m6) = (m2 / drawn, m4 / drawn, m6 / drawn);
  // Var[z⁴] = E[z⁸] − E[z⁴]² = 105 − 9 and Var[z⁶] = 10395 − 225, so the
  // standard errors below are the law's own, not a chosen tolerance.
  let kurtosis_se = (96.0_f64 / drawn).sqrt();
  assert!(
    (m4 / (m2 * m2) - 3.0).abs() < 5.0 * kurtosis_se,
    "kurtosis {} against 3 ± {}",
    m4 / (m2 * m2),
    5.0 * kurtosis_se
  );
  let sixth_se = (10_170.0_f64 / drawn).sqrt();
  assert!(
    (m6 / m2.powi(3) - 15.0).abs() < 5.0 * sixth_se,
    "sixth moment {} against 15 ± {}",
    m6 / m2.powi(3),
    5.0 * sixth_se
  );
}
