//! Points beside a law's lower end, where the end's own law holds, against mpmath at 60 and 120 digits; a value below
//! the normal doubles is its correctly rounded double.

use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;

use crate::rel;

/// Below its unit's normal doubles a unit shape's gamma reads no `0·ln 0` in its regular factor; `k` an ulp-scale
/// either side keeps its log term.
#[test]
fn a_unit_gamma_beside_zero() {
  for (k, want) in [
    (1.0, [0.0029134996012173953, 0.0029134996012173953]),
    (1.0 + 1e-12, [0.002913499599032947, 0.002913499599142348]),
    (1.0 - 1e-12, [0.002913499603401601, 0.002913499603292212]),
  ] {
    let law = SimdTruncatedGamma::<f64>::new(k, 343.2298393252409, 0.0, f64::INFINITY);
    for (x, want) in [5e-324, 1e-307].into_iter().zip(want) {
      let got = law.pdf(x).unwrap();
      assert!(
        rel(got, want) < 1e-12,
        "TG({k}): pdf({x:e}) = {got:e}, mpmath {want:e}"
      );
    }
  }
}

/// Below its unit's normal doubles a gamma's cdf beside 0 is the mass from 0 alone, its power of the point scaled once:
/// a clipped end reads its 0, an unclipped one its correctly rounded subnormal; above that the walk's normal value.
#[test]
fn a_gamma_cdf_beside_zero() {
  let inf = f64::INFINITY;
  for (k, theta, up, rows) in [
    (5.0, 1.0, inf, vec![(1e-310, 0.0), (1e-300, 0.0)]),
    // `ln v + k·ln t` and `k·q` past the doubles: the true `ln F` is −1.2e308 or less on both sides.
    (1.25e305, 1.0, inf, vec![(5e-324, 0.0), (1e-310, 0.0)]),
    (1.68e305, 1.0, 10.0, vec![(5e-324, 0.0)]),
    (
      8.98e304,
      1.086657414900513e303,
      3.896229425657342e305,
      vec![(5e-324, 0.0), (1e-300, 0.0)],
    ),
    (
      70.91814633258258,
      4.369422513910083e-2,
      1.6330458733436888,
      vec![(5e-324, 0.0), (2e-323, 0.0)],
    ),
    (
      1.000000001,
      3.526219794001802e-2,
      1.4594968340809855e-1,
      vec![
        (5e-324, 1.43e-322),
        (2e-323, 5.7e-322),
        (1e-322, 2.846e-321),
      ],
    ),
    (
      1.0,
      343.2298393252409,
      inf,
      vec![(1e-320, 3e-323), (1e-307, 2.9134996012174e-310)],
    ),
    (
      1.0 - 1e-12,
      343.2298393252409,
      inf,
      vec![(1e-307, 2.91349960329513e-310)],
    ),
    (
      1.0,
      6.994007798504422e-3,
      1.002908112128621e-2,
      vec![(5e-324, 9.3e-322), (2e-323, 3.71e-321)],
    ),
  ] {
    let law = SimdTruncatedGamma::<f64>::new(k, theta, 0.0, up);
    for (x, want) in rows {
      assert_eq!(law.cdf(x).unwrap(), want, "TG({k}, {theta}): cdf({x:e})");
    }
  }
  // Within 4 subnormal ulps of mpmath's double: every `exp` and `ln` 4ε off moves it at most 1.
  let got = SimdTruncatedGamma::<f64>::new(1.0 + 1e-12, 343.2298393252409, 0.0, inf)
    .cdf(1e-307)
    .unwrap();
  assert!(
    (got - 2.91349959913945e-310).abs() <= 4.0 * 5e-324,
    "TG(1 + 1e-12): cdf(1e-307) = {got:e}"
  );
  let normal = SimdTruncatedGamma::<f64>::new(
    1.000000001,
    3.526219794001802e-2,
    0.0,
    1.4594968340809855e-1,
  );
  // Within 4 of its 5.8e14 subnormal ulps: every `exp` and `ln` 4ε off moves it at most 2.
  let got = normal.cdf(1e-310).unwrap();
  assert!(
    (got - 2.881829657218485e-309).abs() <= 4.0 * 5e-324,
    "cdf(1e-310) = {got:e}"
  );
  let got = normal.cdf(3e-309).unwrap();
  assert!(
    rel(got, 8.645489001060503e-308) < 1e-12,
    "cdf(3e-309) = {got:e}"
  );
}

/// Below the clip a subnormal cdf is its correctly rounded double (mpmath at 60 and 120 digits), each exact value at
/// least 26ε from a rounding midpoint, where a sum of two rounded logs near −700 would leave 1e-13.
#[test]
fn a_subnormal_cdf_below_the_clip() {
  let inf = f64::INFINITY;
  let rows: [(Box<dyn DistributionExt>, f64, f64); 5] = [
    (
      Box::new(SimdTruncatedGamma::<f64>::new(1.5, 1.0, 0.0, inf)),
      1.1541066389166827e-208,
      9.3268009696e-313,
    ),
    (
      Box::new(SimdTruncatedBeta::<f64>::new(1.5, 3.0, 0.0, 1.0)),
      3.595949997637054e-209,
      9.43405760163e-313,
    ),
    (
      Box::new(SimdTruncatedGamma::<f64>::new(2.0, 1.0, 0.0, inf)),
      2.165349055055685e-156,
      2.344368265113e-312,
    ),
    (
      Box::new(SimdTruncatedNormal::<f64>::new(
        -0.6893128380476039,
        14.737783074127448,
        -inf,
        18.819391168100644,
      )),
      -557.4271305288851,
      1.53881440022e-312,
    ),
    (
      Box::new(SimdTruncatedNormal::<f64>::new(
        7.5004838195075045,
        0.032915368466766404,
        5.721988168218167,
        6.239668987712654,
      )),
      5.727166137508286,
      1.53871788152e-312,
    ),
  ];
  for (law, x, want) in rows {
    assert_eq!(law.cdf(x).unwrap(), want, "cdf({x:e})");
  }
}

/// A lower bound far finer than the gamma's scale keeps its own digits in the unit: the cdf beside it and the mean
/// against mpmath at 60 and 120 digits (shape ½ by erf, the tiny shape by its E1 limit).
#[test]
fn a_gamma_lower_bound_far_below_its_scale() {
  let inf = f64::INFINITY;
  let half = SimdTruncatedGamma::<f64>::new(0.5, 347.32, 1.8632181938306767e-308, inf);
  let tiny = SimdTruncatedGamma::<f64>::new(2.17e-308, 1e300, 1e-200, inf);
  for (got, want) in [
    (
      half.cdf(1.8632181938325403e-308).unwrap(),
      4.133183150592195e-168,
    ),
    (tiny.cdf(1.01e-200).unwrap(), 8.647082894057458e-6),
    (tiny.mean().unwrap(), 8.690246607532952e296),
  ] {
    assert!(rel(got, want) < 1e-12, "{got:e} for {want:e}");
  }
}

/// A huge shape's mass against its upper bound: the variance, `1/λ²` of the exponential below it, formed in the law's
/// coordinate before its square rounds (mpmath at 60 and 120 digits); in the unit it would underflow.
#[test]
fn a_gamma_variance_against_its_upper_bound() {
  for (k, theta, up, want) in [
    (
      1e306,
      1.086657414900513e303,
      3.896229425657342e305,
      0.1518060373735814,
    ),
    (
      1e200,
      1.086657414900513e100,
      3.896229425657342e103,
      1.518060373735814e-193,
    ),
  ] {
    let got = SimdTruncatedGamma::<f64>::new(k, theta, 0.0, up)
      .variance()
      .unwrap();
    assert!(rel(got, want) < 1e-12, "TG({k:e}): variance {got:e}");
  }
}

/// Beside the lower end: a regular end's zone (`TG(1, 343.23)`, `TB(1, 0.384)`), a branch point's zone either side of its
/// edge (`TB(1 + 1e-6, 1)`, `TG(1.058, 1)`), the walk beside a regular end, and next to 1 the walk for a tiny `t^e`.
#[test]
fn the_zone_beside_the_lower_end() {
  let inf = f64::INFINITY;
  let exp = SimdTruncatedGamma::<f64>::new(1.0, 343.2298393252409, 0.0, inf);
  assert_eq!(exp.quantile(5e-324).unwrap(), 1.695e-321);
  let unit = SimdTruncatedBeta::<f64>::new(1.0, 0.38424026401475586, 0.0, 1.0);
  assert_eq!(unit.quantile(5e-324).unwrap(), 1.5e-323);
  for (x, want) in [(1.5e-323, 5e-324), (1e-310, 3.8424026401474e-311)] {
    assert_eq!(unit.cdf(x).unwrap(), want, "TB(1, 0.384): cdf({x:e})");
  }
  let near = SimdTruncatedBeta::<f64>::new(1.000001, 1.0, 0.0, 1.0);
  for (x, want) in [
    (5e-324, 5e-324),
    (2e-323, 2e-323),
    (1e-310, 9.9928645331684e-311),
  ] {
    assert_eq!(near.cdf(x).unwrap(), want, "TB(1 + 1e-6, 1): cdf({x:e})");
  }
  let vanishing = SimdTruncatedGamma::<f64>::new(1.058, 1.0, 0.0, inf);
  for (x, want) in [(1e-305, 2e-323), (1e-300, 3.880476e-318)] {
    assert_eq!(vanishing.cdf(x).unwrap(), want, "TG(1.058, 1): cdf({x:e})");
  }
  let rows = [
    (near.cdf(1e-300).unwrap(), 9.993094630026467e-301),
    (
      SimdTruncatedBeta::<f64>::new(0.5, 3.0, 0.0, 1.0)
        .cdf(1e-271)
        .unwrap(),
      5.929270612815711e-136,
    ),
    (
      SimdTruncatedBeta::<f64>::new(0.5, 3.0, 0.0, 1.0)
        .cdf(1e-269)
        .unwrap(),
      5.929270612815712e-135,
    ),
    (unit.cdf(1.8e-8).unwrap(), 6.916324790594754e-9),
    (unit.cdf(2e-8).unwrap(), 7.684805327615055e-9),
  ];
  for (got, want) in rows {
    assert!(rel(got, want) < 1e-12, "{got:e} against mpmath {want:e}");
  }
  let tiny =
    SimdTruncatedGamma::<f64>::new(1.3613691839394225e-283, 4.3126932660537823e-5, 0.0, inf);
  let next_to_one = tiny.cdf(5.8711677121886804e-288).unwrap();
  assert!(next_to_one <= 1.0 && 1.0 - next_to_one <= 8.0 * f64::EPSILON);
}

/// Below `ZONE_MASS` the quantile beside a regular end takes the end's exponential where the curvature moves `ln k` by
/// under ε, `t < √ε` here, else walks: a normal 32σ below its mode either side of that (mpmath at 60 and 120 digits).
#[test]
fn the_zone_quantile_beside_a_regular_end() {
  let law = SimdTruncatedNormal::<f64>::new(32.0, 1.0, 0.0, f64::INFINITY);
  for (p, want) in [
    (2.270276606003911e-231, 1.3e-8),
    (2.532231659777926e-231, 1.45e-8),
    (2.671941026932039e-231, 1.53e-8),
    (2.968823444009824e-231, 1.7e-8),
  ] {
    let got = law.quantile(p).unwrap();
    assert!(
      rel(got, want) < 1e-12,
      "quantile({p:e}) = {got:e}, mpmath {want:e}"
    );
  }
}

/// A normal whose standardised offset from its lower bound the doubles hold only as a coarse subnormal reads it in the
/// zone, either side of `2⁻¹⁰³²`; a quantile there, or a bound the walk returns, maps back from the bound itself.
#[test]
fn a_normal_beside_its_lower_bound() {
  let (max, inf) = (f64::MAX, f64::INFINITY);
  let steep = SimdTruncatedNormal::<f64>::new(-max, 1.5, 0.0, inf);
  assert_eq!(steep.quantile(5e-324).unwrap(), 0.0);
  for (x, want) in [
    (5e-324, 3.9474596431116665e-16),
    (2e-323, 1.5789838572446656e-15),
    (3.2e-311, 0.002553453502481282),
    (3.3e-311, 0.0026331437770921416),
    (1e-310, 0.007957914071510653),
  ] {
    let got = steep.cdf(x).unwrap();
    assert!(
      rel(got, want) < 1e-12,
      "cdf({x:e}) = {got:e}, mpmath {want:e}"
    );
  }
  for p in [1e-10, 0.9] {
    let x = steep.quantile(p).unwrap();
    let (below, at, above) = (
      steep.cdf(x.next_down()).unwrap(),
      steep.cdf(x).unwrap(),
      steep.cdf(x.next_up()).unwrap(),
    );
    let tol = 1e-12 * p.min(1.0 - p) + 8.0 * f64::EPSILON;
    assert!(
      below.min(at) - tol <= p && p <= above.max(at) + tol,
      "quantile({p})"
    );
  }
  let half = SimdTruncatedNormal::<f64>::new(0.0, 18.05394799735874, 7.313071214234367e-309, inf);
  let x = half.quantile(5e-324).unwrap();
  assert_eq!((x, half.cdf(x).unwrap()), (7.31307121423448e-309, 5e-324));
  let flat = SimdTruncatedNormal::<f64>::new(
    0.0,
    1.0897757579695164e306,
    -1.054893599337657e-308,
    4.3236333773498275e302,
  );
  let x = flat.quantile(5e-324).unwrap();
  assert!(
    rel(x, 2.1361586609201363e-21) < 1e-12,
    "quantile(5e-324) = {x:e}"
  );
}

/// Below the lower clip the cdf's bound `k(x)(x − lo)/∫k` reads the point's exact offset, which a standardised double
/// rounds to 0 beside the bound, and an open end's offset capped at `MAX`.
#[test]
fn below_the_lower_clip_the_bound_reads_the_exact_offset() {
  let clipped = SimdTruncatedNormal::<f64>::new(
    0.0,
    8.751602610882394e158,
    -8.10140922086644e159,
    -4.3051125511004263e148,
  );
  for (x, want) in [
    (-8.101409220866439e159, 3.5090984557242428e-34),
    (-8.1014092208664e159, 9.123655984884914e-33),
  ] {
    let got = clipped.cdf(x).unwrap();
    assert!(
      rel(got, want) < 1e-12,
      "cdf({x:e}) = {got:e}, mpmath {want:e}"
    );
  }
  let open = SimdTruncatedNormal::<f64>::new(
    1.1760943437137064,
    1.5771126610358466,
    f64::NEG_INFINITY,
    -1.402332453578786e124,
  );
  for x in [-1.4023324535787862e124, -1.5e124] {
    assert_eq!(open.cdf(x).unwrap(), 0.0, "cdf({x:e})");
  }
}
