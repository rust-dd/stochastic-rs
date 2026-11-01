//! An open far tail whose clipped piece the budgets would split past the cap: the recut cuts it from the kernel's maximum
//! over it, and the debug suite runs the assertion on the recut piece.

use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;

use crate::rel;

/// `(−∞, −4.72e34]`, 4.6e34 standard deviations below `μ`: a piece of 1.6e36 panels before its recut. The law is the
/// exponential of scale `σ²/|up − μ|` = 2.3e-35 below its bound (mpmath at 60 and 120 digits).
#[test]
fn a_far_open_tail_recuts_its_piece() {
  let up = -4.722020836512684e34;
  let law = SimdTruncatedNormal::<f64>::new(
    -1.4657822581465938,
    1.031724206117544,
    f64::NEG_INFINITY,
    up,
  );
  assert_eq!(law.mean().unwrap(), up);
  assert_eq!(law.median().unwrap(), up);
  assert!(rel(law.variance().unwrap(), 5.081579000827813e-70) < 1e-12);
  assert!((law.skewness().unwrap() + 2.0).abs() < 1e-12);
  assert!((law.kurtosis().unwrap() - 6.0).abs() < 1e-12);
  assert!(rel(law.entropy().unwrap(), -78.77766723467744) < 1e-12);
  assert_eq!(law.cdf(up.next_down()).unwrap(), 0.0);
  assert_eq!(law.quantile(0.5).unwrap(), up);
}
