use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;

use super::Want;
use super::check;
use super::rel;

fn tb(a: f64, b: f64, lo: f64, up: f64) -> SimdTruncatedBeta<f64> {
  SimdTruncatedBeta::<f64>::new(a, b, lo, up)
}

fn want(
  mean: f64,
  variance: f64,
  skewness: f64,
  kurtosis: f64,
  pdf_mid: f64,
  cdf_mid: f64,
) -> Want {
  Want {
    mean,
    variance,
    skewness,
    kurtosis,
    pdf_mid: Some(pdf_mid),
    cdf_mid: Some(cdf_mid),
    median: None,
    entropy: None,
    mgf_1: None,
  }
}

/// The bulk, a far upper tail where `I_b − I_a` keeps few digits, and a concentrated law.
#[test]
fn bulk_and_far_tails() {
  check(
    "TB(2, 3, [0.2, 0.7])",
    &tb(2.0, 3.0, 0.2, 0.7),
    0.45,
    0.5,
    [0.3, 0.45, 0.6],
    &want(
      0.4236573759347383,
      0.018298248346343374,
      0.21090071567353755,
      -1.0259744115311493,
      2.22093813732155,
      0.5822144799456153,
    ),
  );
  check(
    "TB(2, 3, [0.99, 0.999])",
    &tb(2.0, 3.0, 0.99, 0.999),
    0.9945,
    0.009,
    [0.9901, 0.991, 0.995],
    &want(
      0.9924969793517554,
      3.714116848357432e-6,
      0.84147027520417,
      0.02358574657308661,
      91.02451511004718,
      0.833898811347349,
    ),
  );
  check(
    "TB(50, 30, [0.6, 0.65])",
    &tb(50.0, 30.0, 0.6, 0.65),
    0.625,
    0.05,
    [0.61, 0.625, 0.64],
    &want(
      0.6252073618368325,
      0.000202598974309259,
      -0.01728592723814493,
      -1.170930307995844,
      20.691720253998874,
      0.4936682896531081,
    ),
  );
}

/// Widths 1e-2, 1e-4, 1e-6 at 0.4, where a variance from raw moments cancels `(mean/width)²`-fold (references at the
/// f64 bound `0.4 + w`).
#[test]
fn narrow_intervals() {
  for (w, mean, variance, skewness, kurtosis, pdf_mid, cdf_mid) in [
    (
      0.01,
      0.40499256545981566,
      8.332973882797173e-6,
      0.0030904676418322327,
      -1.1999465365327724,
      100.00456267222626,
      0.5011152028238582,
    ),
    (
      0.0001,
      0.4000499993050637,
      8.333333297642436e-10,
      2.8887960496656148e-5,
      -1.1999999948001745,
      10000.000046290646,
      0.5000104240448798,
    ),
    (
      1.0e-6,
      0.40000049999993054,
      8.33333333288384e-14,
      2.886771793685075e-7,
      -1.1999999999994801,
      1000000.0000272184,
      0.5000001041951601,
    ),
  ] {
    check(
      &format!("TB(2, 3, [0.4, 0.4 + {w:e}])"),
      &tb(2.0, 3.0, 0.4, 0.4 + w),
      0.4 + 0.5 * w,
      w,
      [0.4 + 0.25 * w, 0.4 + 0.5 * w, 0.4 + 0.75 * w],
      &want(mean, variance, skewness, kurtosis, pdf_mid, cdf_mid),
    );
  }
}

/// Singular ends at 0 and at 1: a right end's offsets are formed from 1, where `1 − x` would round to 0.
#[test]
fn singular_ends() {
  check(
    "TB(0.5, 0.5, [0, 1e-6])",
    &tb(0.5, 0.5, 0.0, 1e-6),
    5e-7,
    1e-6,
    [1e-7, 5e-7, 9e-7],
    &want(
      3.333333777777989e-7,
      8.888889735450157e-14,
      0.6388763459564561,
      -0.8571432133581712,
      707106.8401121161,
      0.7071067222609525,
    ),
  );
  check(
    "TB(0.5, 0.5, [0.999999, 1])",
    &tb(0.5, 0.5, 0.999999, 1.0),
    0.5 * (0.999999 + 1.0),
    1e-6,
    [0.9999991, 0.9999995, 0.9999999],
    &want(
      0.9999996666666222,
      8.888889735961371e-14,
      -0.6388763459564561,
      -0.8571432133581712,
      707106.8400525304,
      0.29289327769979523,
    ),
  );
  let law = tb(3.0, 0.4, 0.9, 1.0);
  for (what, got, want) in [
    ("mean", law.mean(), 0.9731549993327303),
    ("variance", law.variance(), 0.0008044143848673002),
    ("skewness", law.skewness(), -0.9626166879617979),
    ("kurtosis", law.kurtosis(), -0.26977734352127086),
    ("pdf(0.95)", law.pdf(0.95), 5.793116859448108),
    ("cdf(0.95)", law.cdf(0.95), 0.22021980144267458),
  ] {
    assert!(
      rel(got.unwrap(), want) < 1e-10,
      "TB(3, 0.4, [0.9, 1]): {what} = {got:?}"
    );
  }
}

/// A unit first shape drops the `(a − 1) ln x` term, which reads `0·(−∞)` at 0.
#[test]
fn unit_shape_has_no_log_term() {
  check(
    "TB(1, 3, [0, 0.5])",
    &tb(1.0, 3.0, 0.0, 0.5),
    0.25,
    0.5,
    [0.1, 0.25, 0.4],
    &want(
      0.19642857142857142,
      0.018558673469387756,
      0.44594439866163704,
      -0.8894321040138874,
      1.9285714285714286,
      0.6607142857142857,
    ),
  );
}

/// The whole support is the base law, both ends singular for Beta(½, ½).
#[test]
fn whole_support_is_the_base_law() {
  check(
    "TB(2, 3, [0, 1])",
    &tb(2.0, 3.0, 0.0, 1.0),
    0.5,
    1.0,
    [0.2, 0.5, 0.8],
    &want(
      0.4,
      0.04,
      0.2857142857142857,
      -0.6428571428571429,
      1.5,
      0.6875,
    ),
  );
  check(
    "TB(0.5, 0.5, [0, 1])",
    &tb(0.5, 0.5, 0.0, 1.0),
    0.5,
    1.0,
    [0.1, 0.5, 0.9],
    &want(0.5, 0.125, 0.0, -1.5, std::f64::consts::FRAC_2_PI, 0.5),
  );
}
