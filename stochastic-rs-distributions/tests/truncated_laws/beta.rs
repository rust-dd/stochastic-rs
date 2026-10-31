use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;

use super::NEXT_TO_ONE;
use super::Want;
use super::check;
use super::moments_match;
use super::quantile_is_monotone;
use super::rel;
use super::singular_row;

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
      0.49366828965310805,
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
      -1.1999465365327726,
      100.00456267222626,
      0.5011152028238581,
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

/// Sums in units of the clipped width: an interval of 1e-70 keeps every moment (mpmath, 60 and 120 digits).
#[test]
fn moments_on_a_tiny_interval() {
  moments_match(
    "TB(2, 3, [0, 1e-70])",
    &tb(2.0, 3.0, 0.0, 1e-70),
    [
      6.666666666666666e-71,
      5.555555555555555e-142,
      -0.565685424949238,
      -0.6,
    ],
  );
}

/// Levels next to 1 subtract the mass past the upper clip, and the least level walks the deep range relative to its
/// start's kernel value (mpmath, 60 and 120 digits).
#[test]
fn quantile_tails_keep_their_digits() {
  let law = tb(50.0, 30.0, 0.0, 1.0);
  for (p, want) in NEXT_TO_ONE
    .into_iter()
    .zip([0.9388210716845806, 0.9134908997128216])
  {
    let got = law.quantile(p).unwrap();
    assert!(
      rel(got, want) < 1e-12,
      "TB(50, 30, [0, 1]): quantile({p}) = {got}"
    );
  }
  let got = law.quantile(5e-324).unwrap();
  assert!(
    rel(got, 1.2689229249447996e-7) < 1e-12,
    "TB(50, 30, [0, 1]): quantile(5e-324) = {got}"
  );
}

/// First shapes down to 1e-4, whose mass sits almost wholly in the innermost panel beside 0: in `u = (t/h)^a` that
/// panel is exact, and so is every analytic (mpmath, 60 and 120 digits).
#[test]
fn small_shapes_keep_their_digits() {
  for (a, want) in [
    (
      0.0001,
      [
        4.9997500124993755e-5,
        1.666444464350355e-5,
        122.46734285361164,
        17996.790090449034,
        -9990.789495151635,
        0.9550880852800381,
        0.9955049578932621,
        0.9999806842185378,
      ],
    ),
    (
      0.001,
      [
        0.0004997501249375312,
        0.0001664446433689291,
        38.707242941809426,
        1796.7909044034732,
        -993.0906014872606,
        0.6315883018246734,
        0.9559475786074574,
        0.9998067465369478,
      ],
    ),
    (
      0.005,
      [
        0.0024937655860349127,
        0.0008278025687976109,
        17.270010970088705,
        356.7945200866549,
        -194.69350029702773,
        0.10049999999999999,
        0.7982998758979029,
        0.9990316134849375,
      ],
    ),
    (
      0.01,
      [
        0.004975124378109453,
        0.0016446420317381375,
        12.17606476374401,
        176.79903534576732,
        -95.37854894773321,
        0.0101,
        0.6372669179249951,
        0.998057957914221,
      ],
    ),
    (
      0.03,
      [
        0.014778325123152709,
        0.004805256181421486,
        6.947632926798321,
        56.817048083706204,
        -29.778913447046563,
        1.0300000000000005e-6,
        0.2587243024454868,
        0.9941116020507308,
      ],
    ),
    (
      0.06,
      [
        0.029126213592233007,
        0.009241136364056416,
        4.8258718178380375,
        26.843922194746785,
        -13.760331355358995,
        1.0600000000000011e-12,
        0.06688147851490049,
        0.9880420429050223,
      ],
    ),
  ] {
    singular_row(
      &format!("TB({a:e}, 2, [0, 1])"),
      &tb(a, 2.0, 0.0, 1.0),
      want,
    );
  }
}

/// Ends a hair from both singular points step the reference point off them, and a second shape of 1e6 puts it inside
/// the mass beside 0, an e-fold `1/(α + β)` wide (mpmath, 60 and 120 digits).
#[test]
fn ends_beside_the_singular_points() {
  moments_match(
    "TB(½, ½, [1e-300, 1 − 2⁻⁵³])",
    &tb(0.5, 0.5, 1e-300, 1.0 - f64::EPSILON / 2.0),
    [
      0.4999999966460603,
      0.12499999916151508,
      9.486373815423206e-9,
      -1.4999999966460602,
    ],
  );
  moments_match(
    "TB(½, 1e6, [0, 1])",
    &tb(0.5, 1e6, 0.0, 1.0),
    [
      4.99999750000125e-7,
      4.9999875000225e-13,
      2.8284207607992133,
      11.999928000296999,
    ],
  );
}

/// Beside a singular end the quantile solves in `u`, so its tolerance is relative: `sin²(πp/2)` down to the least
/// level, where it underflows to 0, monotone all the way, and the cdf deep beside 0 (mpmath, 60 and 120 digits).
#[test]
fn quantile_and_cdf_beside_a_singular_end() {
  let arcsine = tb(0.5, 0.5, 0.0, 1.0);
  for (p, want) in [
    (1e-8, 2.4674011002723397e-16),
    (1e-12, 2.4674011002723395e-24),
    (1e-30, 2.4674011002723401e-60),
    (1e-100, 2.4674011002723397e-200),
  ] {
    let got = arcsine.quantile(p).unwrap();
    assert!(
      rel(got, want) < 1e-12,
      "TB(½, ½, [0, 1]): quantile({p:e}) = {got:e}"
    );
  }
  assert_eq!(arcsine.quantile(5e-324), Some(0.0));
  quantile_is_monotone("TB(½, ½, [0, 1])", &arcsine);
  for (law, x, want) in [
    (&arcsine, 1e-300, 6.366197723675813e-151),
    (&tb(0.3, 2.0, 0.0, 0.5), 1e-270, 1.3917284672594802e-81),
  ] {
    let got = law.cdf(x).unwrap();
    assert!(rel(got, want) < 1e-12, "cdf({x:e}) = {got:e}");
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
      0.4459443986616371,
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
