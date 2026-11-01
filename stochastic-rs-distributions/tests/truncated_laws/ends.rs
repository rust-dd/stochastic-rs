//! The cdf and the quantile beside an interval's ends keep the relative digits of the tail there, against mpmath at 60
//! and 120 digits.

use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;

use crate::close;

/// `(argument, cdf, quantile at that level)` triples whose names say which.
fn cdfs(name: &str, law: &impl DistributionExt, rows: &[(f64, f64)]) {
  for &(x, want) in rows {
    close(
      name,
      &format!("cdf({x:e})"),
      law.cdf(x).unwrap(),
      want,
      1e-12,
      0.0,
    );
  }
}

fn quantiles(name: &str, law: &impl DistributionExt, rows: &[(f64, f64)]) {
  for &(p, want) in rows {
    close(
      name,
      &format!("quantile({p:e})"),
      law.quantile(p).unwrap(),
      want,
      1e-12,
      0.0,
    );
  }
}

/// A kernel flat across its interval takes the midpoint as its reference point: a partial mass beside an end is a
/// width of its own, not a difference of two offsets from the midpoint.
#[test]
fn beside_the_lower_end_of_a_flat_kernel() {
  let law = SimdTruncatedNormal::<f64>::new(0.0, 1.0, 0.0, 1.4);
  let name = "TN(0, 1, [0, 1.4])";
  cdfs(
    name,
    &law,
    &[
      (1e-12, 9.515769044114258e-13),
      (1e-10, 9.515769044114259e-11),
      (1e-8, 9.515769044114258e-9),
    ],
  );
  quantiles(
    name,
    &law,
    &[
      (1e-12, 1.050887211915389e-12),
      (1e-10, 1.050887211915389e-10),
    ],
  );
  let law = SimdTruncatedNormal::<f64>::new(
    -2.385686929531167e-78,
    1.3690513203790816e32,
    -2.385686929531167e-78,
    1.910982559527912e32,
  );
  quantiles(
    "TN(−2.4e-78, 1.37e32, [μ, 1.91e32])",
    &law,
    &[(1e-10, 1.4365770544985043e22)],
  );
  let law = SimdTruncatedGamma::<f64>::new(1.5, 2.0, 0.1, 0.5);
  let name = "TG(1.5, 2, [0.1, 0.5])";
  cdfs(name, &law, &[(0.100000000001, 1.6450920635975962e-12)]);
  quantiles(name, &law, &[(1e-12, 0.10000000000060787)]);
  let law = SimdTruncatedBeta::<f64>::new(1.5, 1.5, 0.1, 0.5);
  let name = "TB(1.5, 1.5, [0.1, 0.5])";
  cdfs(name, &law, &[(0.100000000001, 1.7053850420779522e-12)]);
  quantiles(name, &law, &[(1e-12, 0.10000000000058638)]);
}
