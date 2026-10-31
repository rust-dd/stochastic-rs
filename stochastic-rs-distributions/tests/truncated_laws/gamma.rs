use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;

use super::Want;
use super::check;

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

/// The bulk, a far upper tail where `P(k, b) − P(k, a)` keeps no digit, one past the mass's underflow, and a shape
/// of 1e4.
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
      1727.803774425042,
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
      100.0005224097927,
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
      -1.1999999999997397,
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
