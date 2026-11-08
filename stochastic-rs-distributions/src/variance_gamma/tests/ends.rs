use super::*;
use crate::tests::holds;

/// mpmath at 60 and 120 digits at `ν` within an ulp of MAX, its shape `1/ν` at `2⁻¹⁰²⁴`, and where
/// `x − μ` passes MAX: the law of `X/2`, a subnormal `σ` kept, its power of two in the exponent.
#[test]
fn the_parameters_ends_answer_their_values() {
  let vg = SimdVarianceGamma::<f64>::new;
  let max = f64::MAX;
  let rows = [
    (vg(1.0, max, 0.0, 0.0).pdf(1.0), 5.562_684_646_268_003e-309),
    (vg(1.0, max, 1.0, 0.0).pdf(5e-324), 1_125_899_906_842_624.1),
    (vg(1.0, 1.0, -max, max).pdf(-max), 7.528_275_021_586_35e-310),
    (
      vg(5e-324, 1.0, -max, max).pdf(-max),
      7.528_275_021_586_35e-310,
    ),
    (
      vg(max, 1e-300, -max, max).pdf(-max),
      1.346_006_834_128_985e-309,
    ),
  ];
  for (i, (got, want)) in rows.into_iter().enumerate() {
    let got = got.unwrap();
    assert!(holds(got, want, 1e-12), "row {i}: {got:e}, want {want:e}");
  }
}

/// mpmath at 60 and 120 digits: a variance under the normal range from two subnormal terms, summed
/// at `2⁶⁰⁰` so it rounds once to its correctly rounded double, where terms rounded apart do not.
#[test]
fn a_subnormal_variance_rounds_once() {
  let vg = SimdVarianceGamma::<f64>::new;
  for (law, want) in [
    (vg(1.4e-162, 1.0, 1.4e-162, 0.0), 5e-324_f64),
    (vg(3e-162, 0.7, 2e-162, 0.0), 1e-323),
  ] {
    let got = law.variance().unwrap();
    assert_eq!(got.to_bits(), want.to_bits(), "{got:e}, want {want:e}");
  }
}

/// mpmath at 60 and 120 digits where `|x − μ|` passes `2¹⁰²³`, where a two-part sum's `s − a`
/// rounds past MAX at a tie.
#[test]
fn a_far_offset_keeps_its_low_part() {
  let vg = SimdVarianceGamma::<f64>::new;
  let max = f64::MAX;
  let rows = [
    (
      vg(1.0, 0.0198, -max, max).pdf(7.588_735_389_976_133e307),
      4.628_585_387_178_4e-311,
    ),
    (
      vg(max, 2.5, max, max).pdf(2.322_103_711_194_38e307),
      2.022_910_240_442_2e-310,
    ),
  ];
  for (i, (got, want)) in rows.into_iter().enumerate() {
    let got = got.unwrap();
    assert!(holds(got, want, 1e-12), "row {i}: {got:e}, want {want:e}");
  }
}
