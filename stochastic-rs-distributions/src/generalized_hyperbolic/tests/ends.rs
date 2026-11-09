use super::*;
use crate::tests::holds;

/// mpmath at 60 and 120 digits where `δγ` is `5e-324`, `|λ|` or `δγ` nears MAX, `x − μ` or `μt`
/// passes it, or a small tilt meets a heavy clock tail: the number, or `+∞` past MAX.
#[test]
fn the_parameters_ends_answer_their_values() {
  let gh = SimdGeneralizedHyperbolic::<f64>::new;
  let (dmin, rmax, max, inf) = (
    2.225_073_858_507_201e-162,
    1.340_780_792_994_259_6e154,
    f64::MAX,
    f64::INFINITY,
  );
  let rows = [
    (gh(0.3, dmin, 0.0, dmin, 0.0).variance(), inf, 0.0),
    (gh(0.3, dmin, 0.0, dmin, 0.0).kurtosis(), 10.0, 1e-12),
    (
      gh(-8.99e307, 1.0, 0.5, 1.0, 0.0).mean(),
      2.780_867_630_700_78e-309,
      1e-12,
    ),
    (
      gh(-8.99e307, 1.0, 0.5, 1.0, 0.0).variance(),
      5.561_735_261_401_56e-309,
      1e-12,
    ),
    (gh(-2.5, 1.0, 0.5, 1e-110, 0.0).kurtosis(), 6.0, 1e-12),
    (gh(1e300, 1.0, 0.0, dmin, 0.0).variance(), 2e300, 1e-12),
    (
      gh(-max, 1.0, 0.0, rmax, 0.0).pdf(0.0),
      0.564_189_583_547_756_3,
      1e-12,
    ),
    (
      gh(max, rmax, 0.0, rmax, 0.0).pdf(0.0),
      0.256_756_958_907_409_5,
      1e-12,
    ),
    (
      gh(1e300, dmin, 0.0, dmin, -max).pdf(max),
      6.276_816_463_7e-313,
      1e-12,
    ),
    (
      gh(-max, 1.0, 0.0, rmax, 0.0).moment_generating_function(0.5),
      1.064_494_458_917_859_4,
      1e-12,
    ),
    (
      gh(max, 1.0, -0.999_999, 1.0, -max).moment_generating_function(1.999_998_999_999_999_9),
      inf,
      0.0,
    ),
    (
      gh(-max, 2.5, 1.5, rmax, -max).moment_generating_function(-3.999_997_5),
      inf,
      0.0,
    ),
    (
      gh(-1e-310, 1e-150, 0.0, 1e-150, 0.0).moment_generating_function(1e-150),
      7.237_026_788_899_43e306,
      1e-12,
    ),
  ];
  for (i, (got, want, rel)) in rows.into_iter().enumerate() {
    let got = got.unwrap();
    assert!(holds(got, want, rel), "row {i}: {got:e}, want {want:e}");
  }
}

/// mpmath at 60 and 120 digits at `λ = MAX`, `μ = −MAX`: the mean where `βE W` passes MAX, and the
/// density at the mean, a double, from the normal clock.
#[test]
fn a_clock_past_max_meets_its_shift() {
  let gh = SimdGeneralizedHyperbolic::<f64>::new;
  let max = f64::MAX;
  let rows = [
    (
      gh(max, 1.0, 0.5, 1.0, -max).mean(),
      5.992_310_449_541_053e307,
    ),
    (
      gh(max, 2.5, 1.5, 1.0, -max).pdf(-0.25 * max),
      2.886_608_037_231_802_4e-155,
    ),
  ];
  for (i, (got, want)) in rows.into_iter().enumerate() {
    let got = got.unwrap();
    assert!(holds(got, want, 1e-12), "row {i}: {got:e}, want {want:e}");
  }
}

/// mpmath at 60 and 120 digits, `β ≠ 0`: the variance where `E W/s` leaves the doubles, the
/// skewness's `(k₁ + Bk₂)^{3/2}` under them, and `B = β²sρ` under them.
#[test]
fn a_tiny_clock_keeps_its_tilts_terms() {
  let gh = SimdGeneralizedHyperbolic::<f64>::new;
  let (narrow, tiny) = (1.154_700_538_379_251_6e-150, 5e-101);
  let rows = [
    (
      gh(-1.0, 2.3e-162, 1.15e-162, 2.3e-162, 0.0).variance(),
      3.943e-321,
    ),
    (
      gh(-1.0, 1e-155, 5e-156, 1e-160, 0.0).variance(),
      7.259_074e-318,
    ),
    (
      gh(-1.0, 1e-100, tiny, narrow, 0.0).skewness(),
      1.530_978_302_140_685_6e246,
    ),
    (
      gh(-1.5, 1e-100, tiny, narrow, 0.0).skewness(),
      1.924_500_897_298_752_5,
    ),
    (
      gh(-1.5, 1e-100, tiny, narrow, 0.0).kurtosis(),
      5.333_333_333_333_333e250,
    ),
  ];
  for (i, (got, want)) in rows.into_iter().enumerate() {
    let got = got.unwrap();
    assert!(holds(got, want, 1e-12), "row {i}: {got:e}, want {want:e}");
  }
}

/// Exact references `μ + β(2λ/γ² + δ²/2(λ − 1))`: `βE W` a third of an ulp under MAX, and the same
/// tilt brought back by `μ = ∓MAX`, whose sum keeps its digits.
#[test]
fn a_tilt_beside_max_keeps_its_digits() {
  let gh = SimdGeneralizedHyperbolic::<f64>::new;
  let lambda = 1.348_269_851_146_736_7e308;
  let rows = [
    (gh(lambda, 1.0, 0.5, 1.0, 0.0), f64::MAX),
    (gh(lambda, 1.0, -0.5, 1.0, 0.0), -f64::MAX),
    (
      gh(lambda, 1.0, -0.5, 1.0, f64::MAX),
      6.652_801_031_782_399e291,
    ),
    (
      gh(lambda, 1.0, 0.5, 1.0, -f64::MAX),
      -6.652_801_031_782_399e291,
    ),
  ];
  for (i, (law, want)) in rows.into_iter().enumerate() {
    let got = law.mean().unwrap();
    assert!(holds(got, want, 1e-12), "row {i}: {got:e}, want {want:e}");
  }
}

/// Exact references `μ + β(2λ/γ² + δ²/2(λ − 1))` at `|β| < ¼`, where `E W` passes `4·MAX` and `μ`
/// may bring the sum back: a finite truth reads finite.
#[test]
fn a_small_tilt_beside_max_reads_its_finite_truth() {
  let gh = SimdGeneralizedHyperbolic::<f64>::new;
  let lambda = 9.437_888_958_027_155e307;
  let rows = [
    (gh(lambda, 0.5, 0.2, 1.0, 0.0), 1.797_693_134_862_315_3e308),
    (
      gh(lambda, 0.5, -0.2, 1.0, 1e300),
      -1.797_693_124_862_315_3e308,
    ),
    (
      gh(lambda, 0.5, 0.2, 1.0, -f64::MAX),
      -3.468_960_538_000_823e292,
    ),
    (
      gh(4.991_834_296_885_676_5e306, 0.0256, 0.01, 1.0, -1e300),
      1.797_693_124_862_315e308,
    ),
    (
      gh(
        6.486_070_917_867_141e172,
        3.42e-136,
        1.94e-136,
        1.05e-119,
        -f64::MAX,
      ),
      1.374_699_357_236_880_8e308,
    ),
  ];
  for (i, (law, want)) in rows.into_iter().enumerate() {
    let got = law.mean().unwrap();
    assert!(holds(got, want, 1e-12), "row {i}: {got:e}, want {want:e}");
  }
}

/// Exact in rationals: `μ + 2βλ/((α − β)(α + β))` lies `2⁻¹⁰³` of itself past `MAX + ½ ulp`, inside
/// the `2⁻¹⁰²` band of the two-part sum from order 50, so it reads MAX, not `+∞`.
#[test]
fn a_tilt_inside_the_bands_width_past_the_midpoint_reads_max() {
  let law = SimdGeneralizedHyperbolic::<f64>::new(
    8.988_465_674_311_579e307,
    2_251_799_813_685_248.5,
    2_251_799_813_685_248.0,
    1.0,
    2.993_760_464_302_081e292,
  );
  let got = law.mean().unwrap();
  assert!(holds(got, f64::MAX, 0.0), "{got:e}, want MAX");
}
