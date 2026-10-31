use num_complex::Complex64;
use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedExp;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;

use super::Want;
use super::check;
use super::moments_match;
use super::rel;

fn te(lambda: f64, lo: f64, up: f64) -> SimdTruncatedExp<f64> {
  SimdTruncatedExp::<f64>::new(lambda, lo, up)
}

/// Every regime of `λL`: the bulk, the open tail, `λL` of 1e-9 and 1e-17 either way, and a far interval.
#[test]
fn moments_pdf_cdf_and_quantile() {
  for (lam, lo, up, mid, scale, points, quarter, want) in [
    (
      1.5,
      0.5,
      2.5,
      1.5,
      2.0,
      [1.0, 1.5, 2.0],
      Some(0.6808150577250249),
      Want {
        mean: 1.0618752736841548,
        variance: 0.2238804224362054,
        skewness: 0.993456672060953,
        kurtosis: 0.22022410608263582,
        pdf_mid: Some(0.3522318304464184),
        cdf_mid: Some(0.8175744761936437),
        median: None,
        entropy: None,
        mgf_1: None,
      },
    ),
    (
      1.5,
      0.5,
      f64::INFINITY,
      1.1666666666666665,
      0.6666666666666666,
      [0.6, 1.0, 2.0],
      None,
      Want {
        mean: 1.1666666666666667,
        variance: 0.4444444444444444,
        skewness: 2.0,
        kurtosis: 6.0,
        pdf_mid: None,
        cdf_mid: None,
        median: None,
        entropy: None,
        mgf_1: None,
      },
    ),
    (
      1.0,
      0.0,
      1e-9,
      5.0e-10,
      1.0e-9,
      [2.5e-10, 5.0e-10, 7.5e-10],
      Some(2.4999999990625e-10),
      Want {
        mean: 4.999999999166667e-10,
        variance: 8.333333333333335e-20,
        skewness: 3.464101615137755e-10,
        kurtosis: -1.2,
        pdf_mid: Some(999999999.9999999),
        cdf_mid: Some(0.500000000125),
        median: None,
        entropy: None,
        mgf_1: None,
      },
    ),
    (
      1.0,
      0.0,
      1e-17,
      5.0e-18,
      1.0e-17,
      [2.5e-18, 5.0e-18, 7.5e-18],
      Some(2.5e-18),
      Want {
        mean: 5.0e-18,
        variance: 8.333333333333335e-36,
        skewness: 3.4641016151377547e-18,
        kurtosis: -1.2,
        pdf_mid: Some(1.0e17),
        cdf_mid: Some(0.5),
        median: None,
        entropy: None,
        mgf_1: None,
      },
    ),
    (
      1.0e-17,
      0.0,
      1.0,
      0.5,
      1.0,
      [0.25, 0.5, 0.75],
      Some(0.25),
      Want {
        mean: 0.5,
        variance: 0.08333333333333333,
        skewness: 3.4641016151377547e-18,
        kurtosis: -1.2,
        pdf_mid: Some(1.0),
        cdf_mid: Some(0.5),
        median: None,
        entropy: None,
        mgf_1: None,
      },
    ),
    (
      1.0,
      50.0,
      60.0,
      55.0,
      10.0,
      [52.5, 55.0, 57.5],
      Some(50.28766693925637),
      Want {
        mean: 50.999545980089906,
        variance: 0.9954595947649525,
        skewness: 1.9679797375829402,
        kurtosis: 5.596541543593173,
        pdf_mid: Some(0.006738252915294543),
        cdf_mid: Some(0.9933071490757152),
        median: None,
        entropy: None,
        mgf_1: None,
      },
    ),
  ] {
    let law = te(lam, lo, up);
    let name = format!("TE({lam}, {lo}, {up})");
    check(&name, &law, mid, scale, points, &want);
    if let Some(q) = quarter {
      assert!(
        rel(law.quantile(0.25).unwrap(), q) < 1e-12,
        "{name}: quantile(0.25)"
      );
    }
  }
}

/// Sums in units of the clipped width: a rate of 1e±100 at `λL = 2` keeps every moment (mpmath, 60 and 120 digits).
#[test]
fn moments_at_extreme_rates() {
  for (lam, mean, variance) in [
    (1e-100, 6.869647145006687e99, 2.7593833903368954e199),
    (1e100, 6.869647145006686e-101, 2.7593833903368955e-201),
  ] {
    moments_match(
      &format!("TE({lam:e}, 0, 2/λ)"),
      &te(lam, 0.0, 2.0 / lam),
      [mean, variance, 0.6799794054179359, -0.5494785442199013],
    );
  }
}

/// `E[e^{itX}] = e^{it·lo} λ/(λ − it) · (1 − e^{−(λ−it)L})/(1 − e^{−λL})` by mpmath at `t = f64(0.7)`; a small
/// `|(λ − it)L|` takes the series, and an `e^{−λL}` that underflows the open tail's `1/(1 − it/λ)`, divided scaled.
#[test]
fn characteristic_function_matches_mpmath() {
  for (lam, lo, up, re, im) in [
    (1.5, 0.5, 2.5, 0.7009387261515727, 0.6364122867490704),
    (
      1.5,
      0.5,
      f64::INFINITY,
      0.639980257692839,
      0.6415552610454428,
    ),
    (1.0, 0.0, 1e-9, 1.0, 3.499999999416667e-10),
    (1.0, 0.0, 1e-17, 1.0, 3.5e-18),
    (1e-17, 0.0, 1.0, 0.9203109817681301, 0.3359397324507308),
    (1.0, 50.0, 60.0, -0.405370956932924, -0.7119201677918281),
    (1.0, 0.0, 1e200, 0.6711409395973155, 0.4697986577181208),
    (1.0, 0.0, f64::MAX, 0.6711409395973155, 0.4697986577181208),
    (1e200, 0.0, 1.0, 1.0, 7e-201),
    (
      1e160,
      0.5,
      f64::INFINITY,
      0.9393727128473789,
      0.34289780745545134,
    ),
  ] {
    let cf = te(lam, lo, up).characteristic_function(0.7).unwrap();
    assert!(
      (cf.re - re).abs() <= 1e-12 * re.abs().max(1.0)
        && (cf.im - im).abs() <= 1e-12 * im.abs().max(1e-6),
      "TE({lam}, {lo}, {up}): {cf}"
    );
  }
  assert_eq!(
    te(1.5, 0.5, 2.5).characteristic_function(0.0).unwrap(),
    Complex64::new(1.0, 0.0)
  );
}

/// The phase from the exact turns of `t·lower` (Payne–Hanek past the doubles) and `λ/(λ − it)` by Smith's division, so
/// an inexact or overflowing product and an overflowing `t/λ` or `tL` keep the value (mpmath, 60 and 120 digits).
#[test]
fn characteristic_function_past_the_doubles() {
  for (lam, lo, up, t, re, im) in [
    (
      1.0,
      1e10,
      f64::INFINITY,
      10000000.1,
      -4.5893984495249704e-8,
      8.884673312592694e-8,
    ),
    (
      1.0,
      98765432109876.5,
      f64::INFINITY,
      1234.567,
      0.0007325443436412384,
      -0.0003456577961482882,
    ),
  ] {
    let cf = te(lam, lo, up).characteristic_function(t).unwrap();
    let err = (cf - Complex64::new(re, im)).norm() / Complex64::new(re, im).norm();
    assert!(err < 1e-12, "TE({lam}, {lo}, {up}).cf({t}) = {cf}");
  }
  for (lam, up, t, want) in [
    (1e-308, f64::INFINITY, 1e10, Complex64::new(0.0, 1e-318)),
    (1e-308, f64::INFINITY, -1e10, Complex64::new(0.0, -1e-318)),
    (
      5e-324,
      f64::MAX,
      1e10,
      Complex64::new(5.50705e-319, 6.3475e-319),
    ),
    (
      5e-324,
      f64::MAX,
      -1e10,
      Complex64::new(5.50705e-319, -6.3475e-319),
    ),
  ] {
    assert_eq!(
      te(lam, 0.0, up).characteristic_function(t),
      Some(want),
      "TE({lam:e}, 0, {up:e}).cf({t:e})"
    );
  }
}

/// At an infinite `t` the mgf is its limit, +∞ where the law has mass in `t`'s direction past 0 and 0 otherwise, and
/// the cf its Riemann–Lebesgue limit 0; the gamma and beta have no mgf at any `t`.
#[test]
fn infinite_t_takes_the_limit() {
  let (inf, ninf) = (f64::INFINITY, f64::NEG_INFINITY);
  let exp = te(1e308, 0.0, inf);
  assert_eq!(exp.moment_generating_function(ninf), Some(0.0));
  assert_eq!(exp.moment_generating_function(inf), Some(inf));
  for t in [inf, ninf] {
    assert_eq!(
      exp.characteristic_function(t),
      Some(Complex64::new(0.0, 0.0))
    );
    assert_eq!(
      te(1.5, 0.5, 2.5).characteristic_function(t),
      Some(Complex64::new(0.0, 0.0))
    );
  }
  let normal = |lo: f64, up: f64| SimdTruncatedNormal::<f64>::new(0.0, 1.0, lo, up);
  for (lo, up, at_inf, at_ninf) in [
    (-1.0, 2.0, inf, inf),
    (0.0, 2.0, inf, 0.0),
    (-2.0, 0.0, 0.0, inf),
    (1.0, inf, inf, 0.0),
  ] {
    let law = normal(lo, up);
    assert_eq!(
      law.moment_generating_function(inf),
      Some(at_inf),
      "[{lo}, {up}]"
    );
    assert_eq!(
      law.moment_generating_function(ninf),
      Some(at_ninf),
      "[{lo}, {up}]"
    );
  }
  for t in [inf, ninf] {
    assert_eq!(
      SimdTruncatedGamma::<f64>::new(2.0, 1.0, 0.0, 1.0).moment_generating_function(t),
      None
    );
    assert_eq!(
      SimdTruncatedBeta::<f64>::new(2.0, 3.0, 0.0, 1.0).moment_generating_function(t),
      None
    );
  }
}
