use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::tests::scalar_ks_best_p;
use crate::traits::DistributionSampler;

const SEEDS: [u64; 3] = [2718, 999, 42];

/// The smallest `|sample mean − want|` over the three pinned seeds, at `n` draws.
fn best_mean_gap(law: SimdGev<f64>, n: usize, want: f64) -> f64 {
  SEEDS
    .iter()
    .map(|&seed| {
      let mut xs = vec![0.0; n];
      law.seeded(&Deterministic::new(seed)).fill_slice(&mut xs);
      (xs.iter().sum::<f64>() / n as f64 - want).abs()
    })
    .fold(f64::INFINITY, f64::min)
}

/// Gumbel (ξ = 0): the sample mean sits within 5 SE of `γ` (variance π²/6), best of three seeds.
#[test]
fn gev_gumbel_sample_mean_euler() {
  let n = 30_000;
  let five_se = 5.0 * (std::f64::consts::PI.powi(2) / 6.0 / n as f64).sqrt();
  let gap = best_mean_gap(SimdGev::<f64>::new(0.0, 1.0, 0.0), n, EULER_MASCHERONI);
  assert!(gap < five_se, "best |mean − γ| = {gap}, 5 SE = {five_se}");
}

/// Fréchet at ξ = 0.2, where the variance `(Γ(1 − 2ξ) − Γ(1 − ξ)²)/ξ²` is finite: within 5 SE of `(Γ(1 − ξ) − 1)/ξ`.
#[test]
fn gev_frechet_sample_mean_closed_form() {
  let (xi, n) = (0.2_f64, 30_000);
  let (g1, g2) = (
    crate::special::gamma(1.0 - xi),
    crate::special::gamma(1.0 - 2.0 * xi),
  );
  let five_se = 5.0 * ((g2 - g1 * g1) / (xi * xi) / n as f64).sqrt();
  let gap = best_mean_gap(SimdGev::<f64>::new(0.0, 1.0, xi), n, (g1 - 1.0) / xi);
  assert!(gap < five_se, "best |mean − E| = {gap}, 5 SE = {five_se}");
}

#[test]
fn scalar_sample_matches_cdf() {
  for xi in [0.0, 0.3, -0.3] {
    let d = SimdGev::<f64>::new(0.0, 1.0, xi);
    let best = scalar_ks_best_p(&d, |x| d.cdf(x).unwrap());
    assert!(best > 0.01, "xi = {xi}: best p = {best}");
  }
}

/// PDF integrates to 1 within the support (numerical Riemann).
#[test]
fn gev_pdf_normalised_gumbel() {
  let g = SimdGev::<f64>::new(0.0, 1.0, 0.0);
  let n = 5000usize;
  let lo = -10.0_f64;
  let up = 30.0_f64;
  let h = (up - lo) / n as f64;
  let s: f64 = (0..n)
    .map(|k| g.pdf(lo + (k as f64 + 0.5) * h).unwrap() * h)
    .sum();
  assert!((s - 1.0).abs() < 1e-3, "Gumbel PDF integrates to {s}");
}

/// CDF matches inverse-CDF identity: F(F⁻¹(u)) = u on a grid.
#[test]
fn gev_cdf_inverse_round_trip() {
  let g = SimdGev::<f64>::new(0.0, 1.0, 0.2);
  for u in [0.1_f64, 0.3, 0.5, 0.7, 0.9] {
    // X = μ - σ/ξ · (1 - (-ln U)^{-ξ})
    let m_ln_u = -u.ln();
    let x = -(1.0 / 0.2) * (1.0 - m_ln_u.powf(-0.2));
    let f = g.cdf(x).unwrap();
    assert!((f - u).abs() < 1e-10, "F({x}) = {f}, expected {u}");
  }
}

/// Two streams from one deterministic seed agree draw for draw.
#[test]
fn gev_deterministic_seed_reproduces_stream() {
  let mut a = SimdGev::<f64>::new(0.5, 1.2, 0.3).seeded(&Deterministic::new(7));
  let mut b = SimdGev::<f64>::new(0.5, 1.2, 0.3).seeded(&Deterministic::new(7));
  for _ in 0..256 {
    assert_eq!(a.sample(), b.sample());
  }
}

/// Support edge: Reverse Weibull (ξ < 0) is bounded above by μ - σ/ξ.
#[test]
fn gev_reverse_weibull_bounded_support() {
  let xi = -0.5_f64;
  let g = SimdGev::<f64>::new(0.0, 1.0, xi);
  let (lo, hi) = g.support();
  assert_eq!(lo, f64::NEG_INFINITY);
  assert_eq!(hi, -1.0 / xi); // = 2.0
  for seed in SEEDS {
    let mut stream = g.seeded(&Deterministic::new(seed));
    for _ in 0..2_000 {
      let x = stream.sample();
      assert!(x <= hi + 1e-9, "seed {seed}: sample {x} exceeds bound {hi}");
    }
  }
}

/// mpmath at 50 and 80 digits: `exp_m1(−ξ ln y)/ξ` keeps the digits `(y^{−ξ} − 1)/ξ` cancels next to ξ = 0, and only
/// ξ = 0 itself takes the Gumbel form.
#[test]
fn tiny_shape_keeps_its_digits() {
  let rel = |a: f64, b: f64| ((a - b) / b).abs();
  for (xi, q, c) in [
    (1e-12, 2.2503673273149776, 0.873_423_018_492_880_2),
    (1e-9, 2.250_367_329_844_522, 0.873_423_018_256_706_8),
    (1e-7, 2.2503675805201199, 0.873_422_994_852_127_4),
  ] {
    let g = SimdGev::<f64>::new(0.0, 1.0, xi);
    assert!(
      rel(g.quantile(0.9).unwrap(), q) < 1e-13,
      "xi = {xi}: quantile {}",
      g.quantile(0.9).unwrap()
    );
    assert!(
      rel(g.cdf(2.0).unwrap(), c) < 1e-13,
      "xi = {xi}: cdf {}",
      g.cdf(2.0).unwrap()
    );
  }
  assert!(
    rel(
      SimdGev::<f64>::new(0.0, 1.0, 1e-12).median().unwrap(),
      0.366_512_920_581_731_5
    ) < 1e-13
  );
  let mut near = SimdGev::<f64>::new(0.0, 1.0, 1e-11).seeded(&Deterministic::new(42));
  let mut gumbel = SimdGev::<f64>::new(0.0, 1.0, 0.0).seeded(&Deterministic::new(42));
  for _ in 0..1_000 {
    let (a, b) = (near.sample(), gumbel.sample());
    assert!(
      (a - b).abs() < 1e-9,
      "ξ = 1e-11 draw {a} against the Gumbel draw {b}"
    );
  }
}

/// One exponential of `−(1/ξ + 1) ln t − t^{−1/ξ}`: next to a Fréchet law's lower end the density underflows to 0
/// where `t^{−1/ξ−1}·e^{−t^{−1/ξ}}` reads `∞·0`.
#[test]
fn pdf_near_the_frechet_lower_end_is_zero() {
  assert_eq!(
    SimdGev::<f64>::new(0.0, 1.0, 1e-3).pdf(-999.9999999999),
    Some(0.0)
  );
}

/// mpmath at 80 digits (120 at ξ = 1e-12): the standardised moments through the series below |ξ| = 1/8 and the Γ forms
/// above it.
#[test]
fn moments_across_the_series_switch() {
  let rel = |a: f64, b: f64| ((a - b) / b).abs();
  for (xi, mean, var, skew, kurt) in [
    (
      0.0,
      0.577_215_664_901_532_9,
      1.6449340668482264,
      1.1395470994046487,
      2.4,
    ),
    (
      1e-12,
      0.577_215_664_902_522,
      1.6449340668525295,
      1.1395470994106153,
      2.4000000000289177,
    ),
    (
      1e-8,
      0.5772156747920929,
      1.6449341098789999,
      1.139547159070774,
      2.400_000_289_176_899,
    ),
    (
      1e-3,
      0.578_205_629_358_647_9,
      1.649248889005169,
      1.1455260279346362,
      2.429_070_973_712_183,
    ),
    (
      0.124,
      0.7159492073041078,
      2.427_656_730_219_535,
      2.177_067_217_637_463,
      11.009_671_742_717_35,
    ),
    (
      0.126,
      0.718_491_294_467_578_3,
      2.445_933_538_600_544,
      2.2015714275877115,
      11.325042137302308,
    ),
    (
      0.2,
      0.821_148_568_626_516_9,
      3.3440355622978816,
      3.535_071_604_621_395,
      45.091512125815773,
    ),
    (
      -0.1,
      0.48649230133126816,
      1.310_045_507_346_831,
      0.637_637_133_903_144_4,
      0.570_166_483_567_393_7,
    ),
    (
      -0.3,
      0.341_764_345_645_742_7,
      0.978_463_317_324_624_9,
      -0.068_742_099_420_967_07,
      -0.289_399_162_107_949_7,
    ),
  ] {
    let g = SimdGev::<f64>::new(0.0, 1.0, xi);
    assert!(rel(g.mean().unwrap(), mean) < 1e-13, "xi = {xi}: mean");
    assert!(
      rel(g.variance().unwrap(), var) < 1e-13,
      "xi = {xi}: variance"
    );
    assert!(
      rel(g.skewness().unwrap(), skew) < 1e-12,
      "xi = {xi}: skewness"
    );
    assert!(
      rel(g.kurtosis().unwrap(), kurt) < 1e-12,
      "xi = {xi}: kurtosis"
    );
  }
}
