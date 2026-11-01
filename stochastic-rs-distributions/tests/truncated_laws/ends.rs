//! The cdf and the quantile beside an interval's ends keep the relative digits of the tail there, against mpmath at 60
//! and 120 digits.

use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedExp;
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

/// The cdf at `rows`' points, each the double nearest mpmath's value, and `cdf(quantile(p))` at the two levels next to 1.
fn next_to_one(name: &str, law: &impl DistributionExt, rows: &[(f64, f64)]) {
  for &(x, want) in rows {
    assert_eq!(law.cdf(x).unwrap(), want, "{name}: cdf({x:e})");
  }
  for p in [1.0 - 1e-10, 1.0 - f64::EPSILON / 2.0] {
    let x = law.quantile(p).unwrap();
    assert_eq!(law.cdf(x).unwrap(), p, "{name}: cdf(quantile({p:e}))");
  }
}

/// Past the median the cdf is one less the mass above, so next to 1 it keeps the digits its complement has.
#[test]
fn next_to_one_the_cdf_keeps_its_complement() {
  let inf = f64::INFINITY;
  next_to_one(
    "TN(0, 1, ℝ)",
    &SimdTruncatedNormal::<f64>::new(0.0, 1.0, -inf, inf),
    &[(6.361340889697422, 0.9999999999)],
  );
  next_to_one(
    "TN(0.5, 2, [−1, 2])",
    &SimdTruncatedNormal::<f64>::new(0.5, 2.0, -1.0, 2.0),
    &[(1.9999999996368798, 0.9999999999)],
  );
  next_to_one(
    "TN(0, 1, [0, 1.4])",
    &SimdTruncatedNormal::<f64>::new(0.0, 1.0, 0.0, 1.4),
    &[(1.4 - 1e-12, 0.9999999999996428)],
  );
  next_to_one(
    "TG(2.5, 1.2, [0, ∞))",
    &SimdTruncatedGamma::<f64>::new(2.5, 1.2, 0.0, inf),
    &[
      (33.33743900619891, 0.9999999999),
      (50.517019341912786, 0.9999999999999999),
    ],
  );
  next_to_one(
    "TB(2, 3, [0, 1])",
    &SimdTruncatedBeta::<f64>::new(2.0, 3.0, 0.0, 1.0),
    &[
      (0.9997075768387249, 0.9999999999),
      (0.999996972270482, 0.9999999999999999),
    ],
  );
}

/// The exponential's cdf below `λL = 1` as `d·φ(λd)` over its mass, so a `λd` below the normal doubles keeps its
/// digits, and past the median as one less the mass above; the references are mpmath's, rounded once.
#[test]
fn the_exponentials_cdf_in_both_tails() {
  let rows = [
    (1.5, 0.5, 2.5, 2.499999998727631, 0.9999999999),
    (
      4.933855665380294e-281,
      0.0,
      3.623160885826047e275,
      1.790063323271919e-48,
      5e-324,
    ),
    (
      5.35671572205022e-13,
      0.0,
      8.648310197568668e-2,
      8.648310197568468e-302,
      1e-300,
    ),
    (
      7.1995643161400245e-233,
      5.89435479243321e-94,
      4.516084254166023e217,
      5.894354792435441e-94,
      5e-324,
    ),
  ];
  for (lambda, lo, up, x, want) in rows {
    let law = SimdTruncatedExp::<f64>::new(lambda, lo, up);
    assert_eq!(
      law.cdf(x).unwrap(),
      want,
      "TE({lambda:e}, [{lo:e}, {up:e}]): cdf({x:e})"
    );
  }
}
