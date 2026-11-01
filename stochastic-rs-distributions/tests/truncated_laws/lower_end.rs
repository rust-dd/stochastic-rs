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
