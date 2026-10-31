use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;

use super::NEXT_TO_ONE;
use super::Want;
use super::check;
use super::moments_match;
use super::quantile_is_monotone;
use super::rel;
use super::singular_row;

fn tg(k: f64, theta: f64, lo: f64, up: f64) -> SimdTruncatedGamma<f64> {
  SimdTruncatedGamma::<f64>::new(k, theta, lo, up)
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

/// The bulk, a far upper tail where `P(k, b) − P(k, a)` keeps no digit, one where it rounds to 0, and a shape of 1e4.
#[test]
fn bulk_and_far_tails() {
  check(
    "TG(2.5, 1.2, [0.8, 4])",
    &tg(2.5, 1.2, 0.8, 4.0),
    2.4,
    3.2,
    [1.2, 2.0, 3.2],
    &want(
      2.291669372897881,
      0.7540641177095656,
      0.1625278343279509,
      -1.0503498692519804,
      0.350393073106277,
      0.5578733501729103,
    ),
  );
  check(
    "TG(2.5, 1.2, [45, 60])",
    &tg(2.5, 1.2, 45.0, 60.0),
    52.5,
    15.0,
    [45.01, 45.1, 46.0],
    &want(
      46.24725209894823,
      1.551513854358676,
      1.9869587055990805,
      5.847098355940363,
      0.0019482637636529952,
      0.9975867015782197,
    ),
  );
  check(
    "TG(2.5, 1.2, [600, 700])",
    &tg(2.5, 1.2, 600.0, 700.0),
    650.0,
    100.0,
    [600.01, 600.1, 601.0],
    &want(
      601.2035963892966,
      1.4486269887391108,
      1.9999643580544908,
      5.999715433962147,
      7.517202854402304e-19,
      1.0,
    ),
  );
  check(
    "TG(1e4, 1, [9900, 10050])",
    &tg(10000.0, 1.0, 9900.0, 10050.0),
    9975.0,
    150.0,
    [9950.0, 9980.0, 10020.0],
    &want(
      9979.191872186424,
      1727.8037744250419,
      -0.11270753716079686,
      -1.1031350364572177,
      0.00726288381045982,
      0.45691482687435403,
    ),
  );
}

/// Widths 1e-2, 1e-4, 1e-6 at 0.8, where a variance from raw moments cancels `(mean/width)²`-fold (references at the
/// f64 bound `0.8 + w`).
#[test]
fn narrow_intervals() {
  for (w, mean, variance, skewness, kurtosis, pdf_mid, cdf_mid) in [
    (
      0.01,
      0.8050085835442697,
      8.333224828851941e-6,
      -0.0035680929423097993,
      -1.1999744823608436,
      100.00052240979271,
      0.49871246819860365,
    ),
    (
      0.0001,
      0.800050000867958,
      8.333333322301789e-10,
      -3.6080332580250036e-5,
      -1.1999999973988185,
      10000.0000052454,
      0.4999869806314129,
    ),
    (
      1.0e-6,
      0.8000005000000868,
      8.333333333811492e-14,
      -3.608435123047343e-7,
      -1.19999999999974,
      999999.9999712968,
      0.499999869736302,
    ),
  ] {
    check(
      &format!("TG(2.5, 1.2, [0.8, 0.8 + {w:e}])"),
      &tg(2.5, 1.2, 0.8, 0.8 + w),
      0.8 + 0.5 * w,
      w,
      [0.8 + 0.25 * w, 0.8 + 0.5 * w, 0.8 + 0.75 * w],
      &want(mean, variance, skewness, kurtosis, pdf_mid, cdf_mid),
    );
  }
}

/// Open upper ends and the singular lower end of a shape below 1, graded toward 0.
#[test]
fn open_and_singular_ends() {
  check(
    "TG(2.5, 1.2, [0.8, inf))",
    &tg(2.5, 1.2, 0.8, f64::INFINITY),
    2.0,
    1.2,
    [1.0, 2.0, 5.0],
    &want(
      3.1805597238058754,
      3.38683846233327,
      1.370164534210848,
      2.7142166536610537,
      0.27350494309896894,
      0.30352442085877124,
    ),
  );
  check(
    "TG(30, 0.5, [20, inf))",
    &tg(30.0, 0.5, 20.0, f64::INFINITY),
    20.5,
    0.5,
    [20.1, 21.0, 23.0],
    &want(
      21.40736776195919,
      1.68616105378163,
      1.6319927462998263,
      3.5941733551848998,
      0.4823666416161314,
      0.2793157565490875,
    ),
  );
  check(
    "TG(0.7, 1, [0, 1e-3])",
    &tg(0.7, 1.0, 0.0, 1e-3),
    5e-4,
    1e-3,
    [1e-4, 5e-4, 9e-4],
    &want(
      0.0004116750010763083,
      8.970052428919336e-8,
      0.31887221721747006,
      -1.1564980677870544,
      861.7250127724625,
      0.6156989344620627,
    ),
  );
}

/// Sums in units of the clipped width: a scale of 1e±150 and 1e±200 moves no moment (mpmath, 60 and 120 digits); the
/// variance at 1e-200 underflows and at 1e200 overflows, as the exact value does.
#[test]
fn moments_at_extreme_scales() {
  for (theta, mean, variance) in [
    (1e-200, 2.1679147482758088e-200, 0.0),
    (1e-150, 2.167914748275809e-150, 7.284337949302775e-301),
    (1e150, 2.167914748275809e150, 7.284337949302775e299),
    (1e200, 2.167914748275809e200, f64::INFINITY),
  ] {
    moments_match(
      &format!("TG(2.5, {theta:e}, [0.8θ, 4θ])"),
      &tg(2.5, theta, 0.8 * theta, 4.0 * theta),
      [mean, variance, 0.3190481188042759, -0.9373102227975924],
    );
  }
}

/// A level next to 1 subtracts the mass past the upper clip, open or finite, and the least level walks the deep range
/// relative to its start's kernel value (mpmath, 60 and 120 digits).
#[test]
fn quantile_tails_keep_their_digits() {
  for (name, law, wants) in [
    (
      "TG(30, 0.5, [0, inf))",
      tg(30.0, 0.5, 0.0, f64::INFINITY),
      [49.55566420055422, 42.96477116124054],
    ),
    (
      "TG(2.5, 1.2, [0.8, inf))",
      tg(2.5, 1.2, 0.8, f64::INFINITY),
      [50.60528444554982, 39.23237834482653],
    ),
    (
      "TG(2.5, 1.2, [0.8, 200])",
      tg(2.5, 1.2, 0.8, 200.0),
      [50.60528444554982, 39.23237834482653],
    ),
  ] {
    for (p, want) in NEXT_TO_ONE.into_iter().zip(wants) {
      let got = law.quantile(p).unwrap();
      assert!(rel(got, want) < 1e-12, "{name}: quantile({p}) = {got}");
    }
  }
  let got = tg(30.0, 0.5, 0.0, f64::INFINITY).quantile(5e-324).unwrap();
  assert!(
    rel(got, 1.0066646743913833e-10) < 1e-12,
    "TG(30, 0.5, [0, inf)): quantile(5e-324) = {got}"
  );
}

/// Shapes down to 1e-4, whose mass sits almost wholly in the innermost panel beside 0: in `u = (t/h)^k` that panel is
/// exact, and so is every analytic (mpmath, 60 and 120 digits).
#[test]
fn small_shapes_keep_their_digits() {
  for (k, want) in [
    (
      0.0001,
      [
        6.320912545111371e-5,
        2.6420576421232246e-5,
        118.22824828313776,
        16313.105239874927,
        -9990.586081250514,
        0.9550686582416881,
        0.9954847087525884,
        0.9999659595983126,
      ],
    ),
    (
      0.001,
      [
        0.0006318276006158304,
        0.0002638878227153765,
        37.36284552072828,
        1628.0443628985454,
        -992.8870664149445,
        0.6314598029360329,
        0.9557530876691439,
        0.9996594669172038,
      ],
    ),
    (
      0.005,
      [
        0.003153288273779755,
        0.0013123997619908516,
        16.66105312260243,
        322.70922365397416,
        -194.48943389969932,
        0.10039765855573454,
        0.7974869489102778,
        0.9982944819428411,
      ],
    ),
    (
      0.01,
      [
        0.006291994633251167,
        0.0026073200163699846,
        11.738610948138804,
        159.5464741103446,
        -95.17383482480531,
        0.01007940360826197,
        0.635967373461305,
        0.99658190312462,
      ],
    ),
    (
      0.03,
      [
        0.018702409736796977,
        0.007616111635734828,
        6.679061763213313,
        50.78669462660723,
        -29.571786783434792,
        1.0236678105946876e-6,
        0.25713372838059134,
        0.9896633151629415,
      ],
    ),
    (
      0.06,
      [
        0.03689249183162857,
        0.014637477219608078,
        4.618622387247142,
        23.61994747950897,
        -13.550097833090646,
        1.0468786592508845e-12,
        0.06605357788339226,
        0.979093155602347,
      ],
    ),
  ] {
    singular_row(
      &format!("TG({k:e}, 1, [0, 1])"),
      &tg(k, 1.0, 0.0, 1.0),
      want,
    );
  }
}

/// An end within a tenth of a scale of the singular point is treated as the point itself: the reference point steps
/// off it, so the power law's tiny sliver is not mistaken for the mass (mpmath, 60 and 120 digits).
#[test]
fn an_end_beside_the_singular_point() {
  for (name, lo, k, want) in [
    (
      "TG(0.1, 1, [1e-25, 1])",
      1e-25,
      0.1,
      [
        0.06058112879372996,
        0.023208516248972647,
        3.4630133987506295,
        12.723771511569883,
      ],
    ),
    (
      "TG(0.7, 1, [1e-300, 1])",
      1e-300,
      0.7,
      [
        0.32767637518556353,
        0.07735440614627132,
        0.7070784932495472,
        -0.6281962855616962,
      ],
    ),
  ] {
    moments_match(name, &tg(k, 1.0, lo, 1.0), want);
  }
}

/// Beside a singular end the quantile solves in `u`, so its tolerance is relative: `P⁻¹(0.7, p)` down to the least
/// level, where it underflows to 0, monotone all the way, and the cdf at 1e-300 (mpmath, 60 and 120 digits).
#[test]
fn quantile_and_cdf_beside_a_singular_end() {
  let open = tg(0.7, 1.0, 0.0, f64::INFINITY);
  for (p, want) in [
    (1e-8, 3.250779361498167e-12),
    (1e-12, 6.276272330333046e-18),
    (1e-30, 1.2117584734026752e-43),
    (1e-100, 1.2117584734026573e-143),
  ] {
    let got = open.quantile(p).unwrap();
    assert!(
      rel(got, want) < 1e-12,
      "TG(0.7, 1, [0, inf)): quantile({p:e}) = {got:e}"
    );
  }
  assert_eq!(open.quantile(5e-324), Some(0.0));
  let short = tg(0.7, 1.0, 0.0, 1e-3);
  for (p, want) in [
    (1e-30, 1.3886784734131294e-46),
    (1e-100, 1.388678473413109e-146),
  ] {
    let got = short.quantile(p).unwrap();
    assert!(
      rel(got, want) < 1e-12,
      "TG(0.7, 1, [0, 1e-3]): quantile({p:e}) = {got:e}"
    );
  }
  quantile_is_monotone("TG(0.7, 1, [0, inf))", &open);
  let deep = open.cdf(1e-300).unwrap();
  assert!(
    rel(deep, 1.1005474055236995e-210) < 1e-12,
    "TG(0.7, 1, [0, inf)): cdf(1e-300) = {deep:e}"
  );
}

/// A unit shape drops the `(k − 1) ln x` term, which reads `0·(−∞)` at 0.
#[test]
fn unit_shape_has_no_log_term() {
  check(
    "TG(1, 1, [0, 2])",
    &tg(1.0, 1.0, 0.0, 2.0),
    1.0,
    2.0,
    [0.2, 1.0, 1.8],
    &want(
      0.6869647145006686,
      0.27593833903368953,
      0.6799794054179359,
      -0.5494785442199013,
      0.4254590641196608,
      0.7310585786300049,
    ),
  );
}

/// The whole support is the base law: exact values that need no quadrature pin the singular-end rule.
#[test]
fn whole_support_is_the_base_law() {
  check(
    "TG(0.7, 1, [0, inf))",
    &tg(0.7, 1.0, 0.0, f64::INFINITY),
    1.0,
    1.0,
    [0.1, 0.5, 2.0],
    &want(
      0.7,
      0.7,
      2.3904572186687876,
      8.571428571428571,
      0.2834081351687088,
      0.7611876235625861,
    ),
  );
  let tiny = tg(0.1, 2.0, 0.0, f64::INFINITY);
  assert!(
    (tiny.mean().unwrap() - 0.2).abs() < 2e-13 && (tiny.variance().unwrap() - 0.4).abs() < 4e-13
  );
  assert!(
    (tiny.skewness().unwrap() - 2.0 / 0.1_f64.sqrt()).abs() < 1e-11
      && (tiny.kurtosis().unwrap() - 60.0).abs() < 6e-10
  );
}
