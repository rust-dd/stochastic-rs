use super::*;

/// mpmath at 60 and 120 digits where `δγ` is subnormal: the clock from `δ` and `√(α − β)√(α + β)`,
/// its ratios at `ln(2/δγ)` from the roots, where the squared clock carries its subnormal rounding.
#[test]
fn a_subnormal_clock_keeps_its_digits() {
  let gh = SimdGeneralizedHyperbolic::<f64>::new;
  let rows = [
    (
      gh(0.5, 2.3e-162, 1.15e-162, 1e-154, 0.0).mean(),
      2.898_550_724_637_681_4e161,
    ),
    (
      gh(0.5, 2.3e-162, 1.15e-162, 1e-154, 0.0).skewness(),
      2.325_510_696_599_781_3,
    ),
    (gh(0.5, 2.3e-162, 1.15e-162, 1e-154, 0.0).kurtosis(), 9.84),
    (gh(5.0, 3e-162, 1e-162, 1e-154, 0.0).mean(), 1.25e162),
    (gh(5.0, 3e-162, 1e-162, 1e-154, 0.0).skewness(), 0.56),
    (gh(5.0, 3e-162, 1e-162, 1e-154, 0.0).kurtosis(), 0.816),
    (gh(0.5, 3e-162, 1e-162, 2.3e-162, 0.0).mean(), 1.25e161),
    (
      gh(0.5, 2.3e-162, 1.15e-162, 2.3e-162, 0.0).mean(),
      2.898_550_724_637_681_4e161,
    ),
    (
      gh(50.5, 2.3e-162, 1.15e-162, 1e-154, 0.0).pdf(0.0),
      4.511_463_221_317_926e-170,
    ),
    (
      gh(50.5, 2.3e-162, 0.0, 1e-154, 0.0).pdf(0.0),
      9.198_639_945_011_667e-164,
    ),
    (
      gh(200.0, 3e-162, 1e-162, 1e-154, 0.0).pdf(0.0),
      3.526_245_112_960_787_5e-174,
    ),
  ];
  for (i, (got, want)) in rows.into_iter().enumerate() {
    let got = got.unwrap();
    assert!(close(got, want, 1e-12), "row {i}: {got:e}, want {want:e}");
  }
}
