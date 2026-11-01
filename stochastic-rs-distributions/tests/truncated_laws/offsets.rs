//! Offsets beside a bound or a clip that no double near a far point holds, against mpmath at 60 and 120 digits.

use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;

use crate::close;

/// A law's name, parameters and `(argument, mpmath value)` rows.
type Law<'a> = (&'a str, [f64; 4], &'a [(f64, f64)]);

fn rows(name: &str, what: &str, f: impl Fn(f64) -> f64, rows: &[(f64, f64)]) {
  for &(x, want) in rows {
    close(name, &format!("{what}({x:e})"), f(x), want, 1e-12, 0.0);
  }
}

/// Below ½ a point's offset from the branch point 1 rounds: a walk from the lower end reads `x − lo` from the two
/// doubles, whether or not `lo − 1` is exact.
#[test]
fn beside_a_lower_end_far_below_the_branch_point_at_one() {
  let laws: [Law; 3] = [
    (
      "TB(1, 3, [0.1, 1])",
      [1.0, 3.0, 0.1, 1.0],
      &[
        (0.100000000001, 3.3333058535552506e-12),
        (0.10000001, 3.333333294541777e-8),
      ],
    ),
    (
      "TB(2, 3, [0.49⁺, 0.5144])",
      [2.0, 3.0, 0.49000000000000005, 0.5144],
      &[(0.490000000001, 4.197937234737055e-11)],
    ),
    (
      "TB(0.7, 5, [0.495, 0.5194])",
      [0.7, 5.0, 0.495, 0.5194],
      &[(0.495000000244, 1.1091828446269998e-8)],
    ),
  ];
  for (name, [a, b, lo, up], cdf) in laws {
    let law = SimdTruncatedBeta::<f64>::new(a, b, lo, up);
    rows(name, "cdf", |x| law.cdf(x).unwrap(), cdf);
  }
  let law = SimdTruncatedBeta::<f64>::new(1.0, 3.0, 0.1, 1.0);
  rows(
    "TB(1, 3, [0.1, 1])",
    "quantile",
    |p| law.quantile(p).unwrap(),
    &[(1e-12, 0.1000000000003)],
  );
}

/// A point beside `lo` by less than half an ulp of its standardised double still lies above it.
#[test]
fn beside_a_lower_end_finer_than_its_standardised_ulp() {
  let laws: [Law; 4] = [
    (
      "TN(−3, 1, [1, 2])",
      [-3.0, 1.0, 1.0, 2.0],
      &[
        (1.0000000000000002, 9.468430003271588e-16),
        (1.0000000000000004, 1.893686000654317e-15),
      ],
    ),
    (
      "TN(−3, 0.5, [0, 2])",
      [-3.0, 0.5, 0.0, 2.0],
      &[(1e-17, 1.2316965209089294e-16)],
    ),
    (
      "TN(−1000, 1.1, [0, ∞))",
      [-1000.0, 1.1, 0.0, f64::INFINITY],
      &[(5e-14, 4.1322364048612e-11)],
    ),
    (
      "TN(−1e10, 3, [−1, 1])",
      [-1e10, 3.0, -1.0, 1.0],
      &[(-0.9999999993761675, 0.500000043674277)],
    ),
  ];
  for (name, [mu, sigma, lo, up], cdf) in laws {
    let law = SimdTruncatedNormal::<f64>::new(mu, sigma, lo, up);
    rows(name, "cdf", |x| law.cdf(x).unwrap(), cdf);
  }
}

/// A quantile walked from a clip held beside a bound maps back from that bound, not through `μ + σz`.
#[test]
fn a_quantile_walked_from_a_clip_maps_back_from_its_bound() {
  let laws: [Law; 4] = [
    (
      "TN(300, 1, (−∞, 0])",
      [300.0, 1.0, f64::NEG_INFINITY, 0.0],
      &[(0.5, -0.0023104560337902686)],
    ),
    (
      "TN(1000, 1.1, (−∞, 0])",
      [1000.0, 1.1, f64::NEG_INFINITY, 0.0],
      &[(0.5, -0.0008387067219307991)],
    ),
    (
      "TN(1e10, 3, [−1, 1])",
      [1e10, 3.0, -1.0, 1.0],
      &[
        (1e-10, 0.9999999792767341),
        (0.1, 0.9999999979276735),
        (0.5, 0.9999999993761676),
      ],
    ),
    (
      "TN(−1e10, 3, [−1, 1])",
      [-1e10, 3.0, -1.0, 1.0],
      &[
        (0.9, -0.9999999979276735),
        (1.0 - 1e-10, -0.9999999792767342),
      ],
    ),
  ];
  for (name, [mu, sigma, lo, up], quantile) in laws {
    let law = SimdTruncatedNormal::<f64>::new(mu, sigma, lo, up);
    rows(name, "quantile", |p| law.quantile(p).unwrap(), quantile);
  }
}
