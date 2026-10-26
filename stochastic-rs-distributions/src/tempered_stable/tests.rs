use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::tests::assert_moments_within;
use crate::tests::scalar_draws;
use crate::traits::DistributionExt;
use crate::traits::DistributionSampler;

fn laplace_transform(xs: &[f64], u: f64) -> f64 {
  xs.iter().map(|x| (-u * x).exp()).sum::<f64>() / xs.len() as f64
}

/// The empirical Laplace transform matches $\exp(\theta(\lambda^\alpha - (u+\lambda)^\alpha))$ from the untilted Kanter
/// limit to a strong tilt, where naive rejection would need $e^{\lambda^\alpha}$ tries.
#[test]
fn laplace_transform_matches_the_closed_form() {
  for (alpha, lambda, theta, seed) in [
    (0.5, 0.0, 1.0, 1u64),
    (0.7, 1.0, 1.0, 2),
    (0.3, 4.0, 2.0, 3),
    (0.9, 30.0, 0.5, 4),
  ] {
    let d = SimdTemperedStable::<f64>::new(alpha, lambda, theta);
    let n = 300_000;
    let mut xs = vec![0.0; n];
    d.seeded(&Deterministic::new(seed)).fill_slice(&mut xs);
    assert!(xs.iter().all(|x| *x > 0.0 && x.is_finite()));
    for u in [0.5, 1.0, 2.0] {
      let want = (theta * (lambda.powf(alpha) - (u + lambda).powf(alpha))).exp();
      let got = laplace_transform(&xs, u);
      assert!(
        (got - want).abs() < 4e-3,
        "α={alpha} λ={lambda} u={u}: {got} vs {want}"
      );
    }
  }
}

/// Cumulant moments: sample mean and variance against κ₁, κ₂.
#[test]
fn sample_moments_match_the_cumulants() {
  let d = SimdTemperedStable::<f64>::new(0.6, 2.0, 1.5);
  let n = 400_000;
  let mut xs = vec![0.0; n];
  d.seeded(&Deterministic::new(9)).fill_slice(&mut xs);
  let mean = xs.iter().sum::<f64>() / n as f64;
  let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
  assert!(
    (mean - d.mean().unwrap()).abs() / d.mean().unwrap() < 0.01,
    "mean {mean} vs {}",
    d.mean().unwrap()
  );
  assert!(
    (var - d.variance().unwrap()).abs() / d.variance().unwrap() < 0.03,
    "var {var} vs {}",
    d.variance().unwrap()
  );
  assert!((d.mean().unwrap() - 1.5 * 0.6 * 2.0_f64.powf(-0.4)).abs() < 1e-14);
  assert!((d.variance().unwrap() - 1.5 * 0.6 * 0.4 * 2.0_f64.powf(-1.4)).abs() < 1e-14);
  assert!(d.skewness().unwrap() > 0.0 && d.kurtosis().unwrap() > 0.0);
  assert!((d.moment_generating_function(0.0).unwrap() - 1.0).abs() < 1e-15);
  assert_eq!(d.moment_generating_function(3.0), Some(f64::INFINITY));
  let cf = d.characteristic_function(0.0).unwrap();
  assert!((cf.re - 1.0).abs() < 1e-15 && cf.im.abs() < 1e-15);
}

#[test]
fn scalar_sample_moments_match_the_cumulants() {
  let d = SimdTemperedStable::<f64>::new(0.6, 2.0, 1.5);
  let xs = scalar_draws(&d, 13, 200_000);
  assert_moments_within(
    &xs,
    d.cumulant(1),
    d.cumulant(2),
    None,
    6.0,
    "tempered stable",
  );
}

#[test]
fn untilted_law_has_infinite_mean() {
  let d = SimdTemperedStable::<f64>::new(0.5, 0.0, 1.0);
  assert_eq!(d.mean().unwrap(), f64::INFINITY);
}

#[test]
fn deterministic_seed_reproduces_stream() {
  let mut a = SimdTemperedStable::<f64>::new(0.7, 1.0, 1.0).seeded(&Deterministic::new(7));
  let mut b = SimdTemperedStable::<f64>::new(0.7, 1.0, 1.0).seeded(&Deterministic::new(7));
  for _ in 0..256 {
    assert_eq!(a.sample(), b.sample());
  }
}

#[test]
#[should_panic(expected = "alpha must satisfy `alpha > 0.0 && alpha < 1.0`")]
fn rejects_alpha_of_one() {
  let _ = SimdTemperedStable::<f64>::new(1.0, 1.0, 1.0);
}

/// `(α, θ, λ)` cells where the plain algorithm overflowed `θ^{1/α}`, underflowed `(√γ + αζ)^{1/α}` or lost `x^{−b}`, with
/// the law's Laplace transform at u = 0.5, 2, 8 and κ₁, κ₂ (mpmath at 50 and 80 digits).
const SMALL_ALPHA_CELLS: [((f64, f64, f64), [f64; 3], f64, f64); 6] = [
  (
    (0.005, 40.0, 1.0),
    [
      0.922_032_065_181_529_7,
      0.802_256_386_163_993_1,
      0.642_834_691_312_355,
    ],
    0.2,
    0.199,
  ),
  (
    (0.002, 4.0, 1.0),
    [
      0.996_760_223_004_088_1,
      0.991_240_033_087_192_8,
      0.982_537_787_299_751_9,
    ],
    0.008,
    0.007984,
  ),
  (
    (0.002, 10.0, 1.0),
    [
      0.991_920_227_169_880_7,
      0.978_243_754_333_641,
      0.956_914_542_254_838_4,
    ],
    0.02,
    0.01996,
  ),
  (
    (0.003, 1.0, 1.0),
    [
      0.998_783_604_975_733_4,
      0.996_704_169_105_766,
      0.993_408_374_081_328_6,
    ],
    0.003,
    0.002991,
  ),
  (
    (0.002, 1.0, 1.0),
    [
      0.999_189_069_872_680_6,
      0.997_802_777_191_592_7,
      0.995_605_565_004_507_5,
    ],
    0.002,
    0.001996,
  ),
  (
    (0.003, 7.0, 1.0),
    [
      0.991_516_243_867_707_8,
      0.977_156_047_355_086_7,
      0.954_761_100_453_796_1,
    ],
    0.021,
    0.020937,
  ),
];

/// Each cell, θ = 1e160 where `θ^{1/α}` overflows at α = ½, and λ = 0 where `(αζ)^{1/α}` underflows, returns 10⁴ draws,
/// none NaN or negative, inside a bound sized for a debug build; λ = 0 passes `f64::MAX` with its tail's 2.9 %.
#[test]
fn small_alpha_and_huge_scale_draws_terminate() {
  let cells: Vec<(f64, f64, f64)> = SMALL_ALPHA_CELLS
    .iter()
    .map(|c| c.0)
    .chain([(0.5, 1e160, 1.0), (0.005, 1.0, 0.0)])
    .collect();
  let count = cells.len();
  let (tx, rx) = std::sync::mpsc::channel();
  std::thread::spawn(move || {
    for (alpha, theta, lambda) in cells {
      let mut xs = vec![0.0; 10_000];
      SimdTemperedStable::<f64>::new(alpha, lambda, theta)
        .seeded(&Deterministic::new(1))
        .fill_slice(&mut xs);
      let bad = xs.iter().filter(|x| x.is_nan() || **x < 0.0).count();
      tx.send((alpha, theta, bad)).unwrap();
    }
  });
  for _ in 0..count {
    let (alpha, theta, bad) = rx
      .recv_timeout(std::time::Duration::from_secs(120))
      .expect("a cell's 10⁴ draws did not return within 120 s");
    assert_eq!(bad, 0, "(α, θ) = ({alpha}, {theta}): NaN or negative draws");
  }
}

/// The empirical Laplace transform at u = 0.5, 2, 8 within 5 SE of the closed form and the mean within 5 SE of κ₁ (SE from
/// κ₂), best of three seeds: the summands `e^{−uX}` are bounded, so the heavy right tail does not widen the test.
#[test]
fn small_alpha_draws_follow_the_law() {
  let n = 10_000;
  for ((alpha, theta, lambda), lt, k1, k2) in SMALL_ALPHA_CELLS {
    let law = SimdTemperedStable::<f64>::new(alpha, lambda, theta);
    let best = [2718u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let mut xs = vec![0.0; n];
        law.seeded(&Deterministic::new(seed)).fill_slice(&mut xs);
        let mut worst = (xs.iter().sum::<f64>() / n as f64 - k1).abs() / (k2 / n as f64).sqrt();
        for (u, want) in [0.5, 2.0, 8.0].into_iter().zip(lt) {
          let e: Vec<f64> = xs.iter().map(|x| (-u * x).exp()).collect();
          let mean = e.iter().sum::<f64>() / n as f64;
          let var = e.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
          worst = worst.max((mean - want).abs() / (var / n as f64).sqrt());
        }
        worst
      })
      .fold(f64::INFINITY, f64::min);
    assert!(
      best < 5.0,
      "(α, θ, λ) = ({alpha}, {theta}, {lambda}): the best seed's worst |z| is {best}"
    );
  }
}

/// At θ = 1e160 and α = ½ the law's relative spread is 1e-80: every draw is κ₁ = θαλ^{α−1} = 5e159 to 1e-12.
#[test]
fn huge_scale_draws_sit_at_the_mean() {
  let mut xs = vec![0.0; 10_000];
  SimdTemperedStable::<f64>::new(0.5, 1.0, 1e160)
    .seeded(&Deterministic::new(1))
    .fill_slice(&mut xs);
  let worst = xs
    .iter()
    .map(|x| ((x - 5e159) / 5e159).abs())
    .fold(0.0, f64::max);
  assert!(worst < 1e-12, "worst relative distance to κ₁: {worst}");
}
