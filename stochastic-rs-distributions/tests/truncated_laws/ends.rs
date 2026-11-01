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
    let name = format!("TE({lambda:e}, [{lo:e}, {up:e}])");
    close(
      &name,
      &format!("cdf({x:e})"),
      law.cdf(x).unwrap(),
      want,
      1e-14,
      0.0,
    );
  }
}

/// Nearer a finite upper bound the exponential's quantile is that bound less its offset, the double nearest mpmath's
/// point, where the cdf returns the level to within its step across that double's ulp.
#[test]
fn the_exponentials_quantile_next_to_one() {
  let rows = [
    (
      6.497601518958696e-46,
      8.048725138150996e42,
      1.4752046733157775e45,
      [1.4752046730704168e45, 1.4752046733157772e45],
    ),
    (
      1.6695394474213777e86,
      0.0,
      1.067811652422418e-287,
      [1.0678116523156369e-287, 1.0678116524224179e-287],
    ),
    (
      5.817367752874037e104,
      0.0,
      2.3578823950611327e-108,
      [2.3578823948251825e-108, 2.3578823950611323e-108],
    ),
    (
      3.7587252230617957e9,
      0.0,
      1.9590086122197923e-11,
      [1.9590086120164988e-11, 1.959008612219792e-11],
    ),
    (1.5, 0.5, 2.5, [2.4999999987276307, 2.4999999999999987]),
  ];
  for (lambda, lo, up, want) in rows {
    let law = SimdTruncatedExp::<f64>::new(lambda, lo, up);
    let name = format!("TE({lambda:e}, [{lo:e}, {up:e}])");
    for (p, x) in [1.0 - 1e-10, 1.0 - f64::EPSILON / 2.0]
      .into_iter()
      .zip(want)
    {
      assert_eq!(law.quantile(p).unwrap(), x, "{name}: quantile({p:e})");
      let step = law.pdf(x).unwrap() * (x.next_up() - x);
      let back = law.cdf(x).unwrap();
      assert!(
        (back - p).abs() <= 1e-12 * (1.0 - p) + step,
        "{name}: cdf at quantile({p:e}) = {back}"
      );
    }
  }
}

/// A lower bound inside a singular end's panel: the solve runs in the offset along `q` from the end the walk starts
/// at, and the point is formed from that end's double, so it keeps its distance from the bound to 1e-12 (or two ulps);
/// mpmath's references from the exponential integral, which these shapes (5e-235, 1e-170) follow far below the bar.
#[test]
fn beside_a_bound_inside_a_singular_panel() {
  let (k, theta, lo) = (
    5.130938376779457e-235,
    2.0685587404340502e123,
    1.3059805401332655e-166,
  );
  let gamma = SimdTruncatedGamma::<f64>::new(k, theta, lo, f64::INFINITY);
  let (a, b, lb) = (
    1.1588573908039842e-170,
    2.971543572469775e18,
    9.489781194495966e-21,
  );
  let beta = SimdTruncatedBeta::<f64>::new(a, b, lb, 1.0);
  let rows: [(&str, &dyn DistributionExt, f64, f64, f64); 6] = [
    ("TG(5.1e-235)", &gamma, lo, 1e-12, 1.3059805410021733e-166),
    ("TG(5.1e-235)", &gamma, lo, 1e-10, 1.3059806270240422e-166),
    ("TG(5.1e-235)", &gamma, lo, 1e-8, 1.3059892292395496e-166),
    ("TG(5.1e-235)", &gamma, lo, 0.9, 1.4793853798408395e94),
    ("TB(1.2e-170)", &beta, lb, 1e-12, 9.489781194525437e-21),
    ("TB(1.2e-170)", &beta, lb, 1e-10, 9.48978119744311e-21),
  ];
  for (name, law, bound, p, want) in rows {
    let bar = (1e-12 * (want - bound) + 2.0 * (want.next_up() - want)).min(1e-12 * want);
    let got = law.quantile(p).unwrap();
    close(name, &format!("quantile({p:e})"), got, want, 0.0, bar);
  }
}
