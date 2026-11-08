use super::*;
use crate::tests::holds;

/// mpmath at 60 and 120 digits where `ω` is `5e-324`, `|λ|` passes MAX/2, `χ` or `ψ` sits at an end
/// or the saddle leaves the doubles: the number, `0` under half a subnormal, `+∞` past MAX.
#[test]
fn the_parameters_ends_answer_their_values() {
  let gig = SimdGig::<f64>::new;
  let (sub, max, inf) = (5e-324, f64::MAX, f64::INFINITY);
  let rows = [
    (gig(0.5, sub, sub).variance(), inf, 1e-12),
    (
      gig(0.5, sub, sub).skewness(),
      2.828_427_124_746_190_1,
      1e-12,
    ),
    (gig(0.5, sub, sub).kurtosis(), 12.0, 1e-12),
    (
      gig(-5.0, sub, sub).skewness(),
      3.464_101_615_137_754_6,
      1e-12,
    ),
    (gig(-5.0, sub, sub).kurtosis(), 42.0, 1e-12),
    (gig(8.99e307, 1.0, 1.0).mean(), inf, 0.0),
    (
      gig(-8.99e307, 1.0, 1.0).mean(),
      5.561_735_261_401_56e-309,
      1e-12,
    ),
    (gig(max, sub, 1e300).mean(), 359_538_626.972_463_12, 1e-12),
    (
      gig(max, sub, 1e300).variance(),
      7.190_772_539_449_262e-292,
      1e-12,
    ),
    (gig(10.0, max, max).mean(), 1.0, 1e-12),
    (gig(-0.5, sub, sub).mean(), 1.0, 1e-12),
    (gig(8.99e307, 1.0, 1.0).pdf(1.0), 0.0, 0.0),
    (gig(0.5, 1e308, 1e308).pdf(1e-300), 0.0, 0.0),
    (gig(1e15, sub, 1.0).pdf(1e300), 0.0, 0.0),
    (
      gig(50.0, max, max).pdf(1.0),
      5.348_941_470_755_712e153,
      1e-12,
    ),
    (
      gig(-50.0, sub, 1.0).pdf(sub),
      1.792_517_692_073_567e245,
      1e-12,
    ),
    (
      gig(50.0, 1.0, 5e-307).pdf(1e308),
      1.801_082_132_6e-312,
      1e-12,
    ),
    (
      gig(0.5, 1e308, 1.0).moment_generating_function(-max),
      0.0,
      0.0,
    ),
    (
      gig(0.5, max, max).moment_generating_function(-1.0),
      0.367_879_441_171_442_32,
      1e-12,
    ),
    (
      gig(-5.5e-309, 1e-300, 1e-300).moment_generating_function(5e-301),
      1.315_823_052_527_165_4e305,
      1e-12,
    ),
  ];
  for (i, (got, want, rel)) in rows.into_iter().enumerate() {
    let got = got.unwrap();
    assert!(holds(got, want, rel), "row {i}: {got:e}, want {want:e}");
  }
}

/// mpmath at 60 and 120 digits at a subnormal `x`, where the two-part sum takes `χ/x`'s residual
/// on the normal grid, within `32ε`; on the subnormal grid it grows like `ε·χ/2x`.
#[test]
fn a_subnormal_argument_keeps_the_spread_in_two_parts() {
  let rows = [
    (
      SimdGig::<f64>::new(
        -5.292_037_462_292_903,
        3.169_890_356_841_680_7e-308,
        6.299_818_254_505_962e-306,
      )
      .pdf(2.053_914_587_194e-311),
      1.821_185_066_402_201_9e-11,
    ),
    (
      SimdGig::<f64>::new(-5.29, 1e-315, 1e-300).pdf(1.428_57e-318),
      5.324_172_009_668_738e177,
    ),
  ];
  for (i, (got, want)) in rows.into_iter().enumerate() {
    let got = got.unwrap();
    assert!(
      holds(got, want, 32.0 * f64::EPSILON),
      "row {i}: {got:e}, want {want:e}"
    );
  }
}

/// mpmath at 60 and 120 digits at a saddle that is a double past `r = 2¹⁰⁰`, and at `x` under
/// `10⁻⁹` of the saddle, where `x/w − 1`'s low part is a share of `x/w`.
#[test]
fn a_concentrated_density_reads_its_saddle_and_far_side() {
  let gig = SimdGig::<f64>::new;
  let (max, min) = (f64::MAX, f64::MIN_POSITIVE);
  let rows = [
    (
      gig(-max, 0.25 * max, 1.0).pdf(0.125),
      4.279_153_176_604_569_5e154,
    ),
    (gig(-1e300, 1e300, 1.0).pdf(0.5), 7.978_845_608_028_654e149),
    (gig(50.0, min, 1e300).pdf(min), 9.270_971_416_144_202e-154),
  ];
  for (i, (got, want)) in rows.into_iter().enumerate() {
    let got = got.unwrap();
    assert!(holds(got, want, 1e-12), "row {i}: {got:e}, want {want:e}");
  }
}

/// mpmath at 60 and 120 digits where `E X/√(E X²)` leaves the normal range (`λ` near −1, `ω`
/// subnormal), and a mean from its ratios' limits, rounded once onto the subnormal grid.
#[test]
fn a_tiny_clock_keeps_its_scale_and_rounds_its_mean_once() {
  let gig = SimdGig::<f64>::new;
  for (i, (got, want)) in [
    (gig(-1.0, 1e-315, 1e-315).variance(), 1.0),
    (gig(-0.99, 5e-324, 5e-324).variance(), 2_931_776.957_579_104),
    (
      gig(-1.001, 1e-312, 1e-312).variance(),
      0.237_628_925_102_080_58,
    ),
  ]
  .into_iter()
  .enumerate()
  {
    let got = got.unwrap();
    assert!(holds(got, want, 1e-12), "row {i}: {got:e}, want {want:e}");
  }
  for (law, want) in [
    (gig(-1.2, 1e-315, 1e-315), 2.5e-315_f64),
    (gig(-3.0, 1e-323, 1e-323), 0.0),
  ] {
    let got = law.mean().unwrap();
    assert_eq!(got.to_bits(), want.to_bits(), "{got:e}, want {want:e}");
  }
}

/// Exact references, `2λ/ψ + χ/2(λ − 1)` and mpmath at order 60: a truth under MAX reads finite and
/// one past `MAX + ½ ulp` reads `+∞`.
#[test]
fn a_mean_beside_max_reads_finite_exactly_where_its_truth_rounds_finite() {
  let gig = SimdGig::<f64>::new;
  let rows = [
    (
      gig(1e305, 1.0, 0.001_112_536_929_253_600_8),
      1.797_693_134_862_315_5e308,
    ),
    (
      gig(5e307, 1.0, 0.556_268_464_626_800_5),
      1.797_693_134_862_315_5e308,
    ),
    (
      gig(60.0, f64::MAX / 4.0, 6.689_362_965_616_521e-307),
      1.797_693_134_862_315_5e308,
    ),
    (gig(1e300, 1.0, 1.112_536_929_253_600_7e-8), f64::INFINITY),
  ];
  for (i, (law, want)) in rows.into_iter().enumerate() {
    let got = law.mean().unwrap();
    assert!(holds(got, want, 1e-12), "row {i}: {got:e}, want {want:e}");
  }
}

/// mpmath at 60 and 120 digits (agreeing to 100) below order 50: a truth under MAX reads finite
/// from the ratios' two parts, and the closed forms at `±½` read `+∞` from `MAX + ½ ulp`.
#[test]
fn a_mean_beside_max_below_order_50_reads_finite_for_a_finite_truth() {
  let gig = SimdGig::<f64>::new;
  let rows = [
    (
      gig(10.0, 4.433_682_798_379_86e307, 1.127_730_653_583_761_8e-307),
      1.797_693_134_862_315_5e308,
      1e-12,
    ),
    (
      gig(2.5, 1.102_037_053_125_984_7e307, 2.835_657_831_227_886e-308),
      f64::MAX,
      1e-12,
    ),
    (
      gig(
        49.0,
        4.482_590_941_754_468_5e307,
        5.465_589_057_388_046_6e-307,
      ),
      f64::MAX,
      1e-12,
    ),
    (
      gig(0.5, 8.988_465_674_311_579e306, 6.953_355_807_835_004e-309),
      f64::INFINITY,
      0.0,
    ),
    (
      gig(-0.5, f64::MAX, 5.562_684_646_268e-309),
      f64::INFINITY,
      0.0,
    ),
  ];
  for (i, (law, want, rel)) in rows.into_iter().enumerate() {
    let got = law.mean().unwrap();
    assert!(holds(got, want, rel), "row {i}: {got:e}, want {want:e}");
  }
}
