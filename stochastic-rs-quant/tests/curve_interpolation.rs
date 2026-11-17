//! The cubic spline and the monotone convex method against independent references: scipy's
//! `CubicSpline(bc_type="natural")` and Hagan & West (2006) §4 with Gauss–Legendre integrals.

use ndarray::array;
use stochastic_rs_quant::curves::BootstrapInstrument;
use stochastic_rs_quant::curves::BootstrapSpec;
use stochastic_rs_quant::curves::CurvePoint;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;
use stochastic_rs_quant::curves::bootstrap;
use stochastic_rs_quant::curves::interpolate_discount_factor;

const CUBIC: InterpolationMethod = InterpolationMethod::CubicSplineOnZeroRates;
const MONOTONE_CONVEX: InterpolationMethod = InterpolationMethod::MonotoneConvex;

fn user(method: InterpolationMethod) -> DiscountCurve<f64> {
  DiscountCurve::from_zero_rates(
    &array![0.5, 1.0, 2.0, 5.0],
    &array![0.03, 0.032, 0.035, 0.04],
    method,
  )
}

fn bootstrapped(
  schedules: &[&[f64]],
  rates: &[f64],
  method: InterpolationMethod,
) -> DiscountCurve<f64> {
  let quotes: Vec<BootstrapInstrument<f64>> = schedules
    .iter()
    .zip(rates)
    .map(|(times, &rate)| BootstrapInstrument::SwapWithSchedule {
      rate,
      fixed_times: times.to_vec(),
      float_times: Vec::new(),
    })
    .collect();
  bootstrap(BootstrapSpec::new(&quotes, method)).unwrap()
}

fn semi_annual(method: InterpolationMethod) -> DiscountCurve<f64> {
  bootstrapped(
    &[
      &[0.5],
      &[0.5, 1.0],
      &[0.5, 1.0, 1.5, 2.0],
      &[1.0, 2.0, 3.0, 4.0, 5.0],
    ],
    &[0.03, 0.031, 0.033, 0.036],
    method,
  )
}

/// Swaps whose payments fall between earlier pillars, so the later pillars depend on the method.
fn paying_between_pillars(method: InterpolationMethod) -> DiscountCurve<f64> {
  bootstrapped(
    &[
      &[0.5],
      &[1.0],
      &[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
      &[0.4, 0.9, 1.4, 1.9, 2.4, 3.0],
    ],
    &[0.03, 0.031, 0.033, 0.035],
    method,
  )
}

fn assert_reads(curve: &DiscountCurve<f64>, expected: &[(f64, f64)]) {
  for &(t, reference) in expected {
    let got = curve.discount_factor(t);
    assert!((got - reference).abs() < 2e-12, "D({t}) = {got}");
  }
}

fn forward(curve: &DiscountCurve<f64>, from: f64, to: f64) -> f64 {
  (curve.discount_factor(from) / curve.discount_factor(to)).ln() / (to - from)
}

#[test]
fn the_cubic_spline_on_zero_rates_is_the_natural_spline() {
  let curve = user(CUBIC);
  assert_reads(
    &curve,
    &[
      (0.75, 0.976_998_286_836),
      (1.5, 0.950_766_846_612),
      (3.0, 0.894_666_417_710),
      (4.0, 0.856_651_034_424),
    ],
  );
  let h = 1e-6;
  let left = (curve.zero_rate(1.0) - curve.zero_rate(1.0 - h)) / h;
  let right = (curve.zero_rate(1.0 + h) - curve.zero_rate(1.0)) / h;
  assert!((left - 0.003_710).abs() < 2e-6 && (right - 0.003_710).abs() < 2e-6);
  assert!(
    (left - right).abs() < 1e-6,
    "one slope at a knot: {left}, {right}"
  );
  // The second derivative vanishes at the last pillar: the zero rate is linear to first order.
  let end =
    (curve.zero_rate(5.0) - 2.0 * curve.zero_rate(5.0 - 1e-3) + curve.zero_rate(5.0 - 2e-3)) / 1e-6;
  assert!(end.abs() < 1e-5, "{end}");
}

#[test]
fn a_bootstrapped_cubic_curve_reads_the_natural_spline_between_its_pillars() {
  assert_reads(
    &semi_annual(CUBIC),
    &[
      (0.1, 0.997_026_706_520),
      (0.75, 0.977_554_603_500),
      (1.5, 0.953_406_943_611),
      (3.0, 0.902_514_360_185),
      (4.0, 0.869_267_281_366),
    ],
  );
  assert_reads(
    &paying_between_pillars(CUBIC),
    &[
      (0.75, 0.977_664_860_193),
      (1.5, 0.953_594_958_620),
      (2.0, 0.936_248_971_992),
      (3.0, 0.900_854_775_352),
    ],
  );
}

#[test]
fn monotone_convex_starts_at_the_origin_as_hagan_and_west_do() {
  assert_reads(
    &semi_annual(MONOTONE_CONVEX),
    &[
      (0.1, 0.997_074_211_487),
      (0.25, 0.992_675_705_709),
      (0.75, 0.977_551_854_151),
      (1.5, 0.953_266_254_070),
      (3.0, 0.903_113_718_814),
      (4.0, 0.869_735_825_063),
    ],
  );
  assert_reads(
    &paying_between_pillars(MONOTONE_CONVEX),
    &[
      (0.25, 0.992_653_311_706),
      (0.75, 0.977_675_712_541),
      (1.5, 0.953_454_176_975),
      (2.0, 0.936_248_587_381),
      (3.0, 0.900_859_585_792),
    ],
  );
  let curve = user(MONOTONE_CONVEX);
  assert_reads(
    &curve,
    &[
      (0.1, 0.997_100_212_529),
      (0.25, 0.992_714_171_277),
      (0.4, 0.988_214_005_433),
      (0.6, 0.981_914_868_180),
      (0.75, 0.976_977_490_446),
      (0.9, 0.971_919_140_599),
      (1.5, 0.950_753_928_672),
      (3.0, 0.894_839_316_814),
    ],
  );
  let (below, above) = (
    forward(&curve, 0.5 - 1e-7, 0.5),
    forward(&curve, 0.5, 0.5 + 1e-7),
  );
  assert!(
    (below - 0.032).abs() < 1e-6 && (above - 0.032).abs() < 1e-6,
    "the forward is continuous at the first pillar: {below}, {above}"
  );
  let one_pillar = DiscountCurve::from_zero_rates(&array![2.0], &array![0.03], MONOTONE_CONVEX);
  assert!((one_pillar.discount_factor(0.7) - (-0.021_f64).exp()).abs() < 1e-15);
}

#[test]
fn a_time_before_the_origin_reads_nan() {
  let points: [CurvePoint<f64>; 3] = [
    CurvePoint {
      time: 0.5,
      discount_factor: 0.985,
    },
    CurvePoint {
      time: 1.0,
      discount_factor: 0.969,
    },
    CurvePoint {
      time: 2.0,
      discount_factor: 0.936,
    },
  ];
  for method in [
    InterpolationMethod::LinearOnZeroRates,
    InterpolationMethod::LogLinearOnDiscountFactors,
    CUBIC,
    MONOTONE_CONVEX,
  ] {
    for (slice, name) in [(&points[..], "three"), (&points[..0], "none")] {
      assert!(
        interpolate_discount_factor(slice, -0.5_f64, method).is_nan(),
        "{method:?}, {name}"
      );
      assert!(interpolate_discount_factor(slice, f64::NAN, method).is_nan());
      assert_eq!(interpolate_discount_factor(slice, 0.0, method), 1.0);
    }
    assert_eq!(interpolate_discount_factor(&points, 0.5, method), 0.985);
  }
}
