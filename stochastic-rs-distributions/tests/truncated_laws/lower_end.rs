//! Points beside a law's lower end, where the end's own law holds, against mpmath at 60 and 120 digits; a value below
//! the normal doubles is its correctly rounded double.

use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;

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
        (1e-310, 2.881829657218485e-309),
      ],
    ),
    (
      1.0,
      343.2298393252409,
      inf,
      vec![(1e-320, 3e-323), (1e-307, 2.9134996012174e-310)],
    ),
    (
      1.0 + 1e-12,
      343.2298393252409,
      inf,
      vec![(1e-307, 2.91349959913945e-310)],
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
  let normal = SimdTruncatedGamma::<f64>::new(
    1.000000001,
    3.526219794001802e-2,
    0.0,
    1.4594968340809855e-1,
  );
  let got = normal.cdf(3e-309).unwrap();
  assert!(
    rel(got, 8.645489001060503e-308) < 1e-12,
    "cdf(3e-309) = {got:e}"
  );
}
