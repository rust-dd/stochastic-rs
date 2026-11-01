//! The cdf and the quantile beside an interval's ends keep the relative digits of the tail there, against mpmath at 60
//! and 120 digits.

use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedExp;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;

use crate::close;

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

/// The cdf at `rows`' points to 8ε, the absolute digits it holds next to 1, and the quantile at 1 − 1e-10 and
/// 1 − 2⁻⁵³ to an ulp of mpmath's `points`: above ½ it walks from the upper end, keeping the digits `1 − p` has.
fn next_to_one(name: &str, law: &impl DistributionExt, rows: &[(f64, f64)], points: [f64; 2]) {
  for &(x, want) in rows {
    let got = law.cdf(x).unwrap();
    assert!(
      (got - want).abs() <= 8.0 * f64::EPSILON,
      "{name}: cdf({x:e}) = {got}, mpmath {want}"
    );
  }
  for (p, want) in [1.0 - 1e-10, 1.0 - f64::EPSILON / 2.0]
    .into_iter()
    .zip(points)
  {
    let got = law.quantile(p).unwrap();
    assert!(
      (got - want).abs() <= want.next_up() - want,
      "{name}: quantile({p:e}) = {got:e}, mpmath {want:e}"
    );
  }
}

#[test]
fn next_to_one_the_quantile_walks_from_the_upper_end() {
  let inf = f64::INFINITY;
  next_to_one(
    "TN(0, 1, ℝ)",
    &SimdTruncatedNormal::<f64>::new(0.0, 1.0, -inf, inf),
    &[(6.361340889697422, 0.9999999999)],
    [6.361340889697422, 8.209536151601387],
  );
  next_to_one(
    "TN(0.5, 2, [−1, 2])",
    &SimdTruncatedNormal::<f64>::new(0.5, 2.0, -1.0, 2.0),
    &[(1.9999999996368798, 0.9999999999)],
    [1.9999999996368798, 1.9999999999999996],
  );
  next_to_one(
    "TN(0, 1, [0, 1.4])",
    &SimdTruncatedNormal::<f64>::new(0.0, 1.0, 0.0, 1.4),
    &[(1.4 - 1e-12, 0.9999999999996428)],
    [1.3999999997199957, 1.3999999999999997],
  );
  next_to_one(
    "TG(2.5, 1.2, [0, ∞))",
    &SimdTruncatedGamma::<f64>::new(2.5, 1.2, 0.0, inf),
    &[
      (33.33743900619891, 0.9999999999),
      (50.517019341912786, 0.9999999999999999),
    ],
    [33.337439006198906, 50.517019341912786],
  );
  next_to_one(
    "TB(2, 3, [0, 1])",
    &SimdTruncatedBeta::<f64>::new(2.0, 3.0, 0.0, 1.0),
    &[
      (0.9997075768387249, 0.9999999999),
      (0.999996972270482, 0.9999999999999999),
    ],
    [0.9997075768387249, 0.999996972270482],
  );
}

/// Next to 1 the cdf at `rows`' points to 8ε of mpmath's value.
fn near_one(rows: &[(&str, &dyn DistributionExt, f64, f64)]) {
  for &(name, law, x, want) in rows {
    let got = law.cdf(x).unwrap();
    assert!(
      (got - want).abs() <= 8.0 * f64::EPSILON,
      "{name}: cdf({x:e}) = {got}, mpmath {want}"
    );
  }
}

/// Next to 1 the cdf sums the panels its total was summed on: on panels laid out afresh up to `x`, these laws, spread
/// over fifty decades, read 20–39 ulps of 1 off (mpmath, from E1).
#[test]
fn next_to_one_the_cdf_walks_the_panels_of_its_total() {
  let (a, b) = (8.419970930322122e-279, 5.5279504620333736e175);
  let first = SimdTruncatedBeta::<f64>::new(a, b, 9.653559279995389e-229, 9.999999999996447e-1);
  let (a, b) = (1.1657359323667004e-158, 1.1822617788814124e192);
  let second = SimdTruncatedBeta::<f64>::new(a, b, 2.44915738197029e-209, 1.0);
  let (k, theta) = (1.1329440812770196e-204, 6.26723461668055e-223);
  let gamma =
    SimdTruncatedGamma::<f64>::new(k, theta, 5.200338863973879e-225, 4.582208590190747e68);
  near_one(&[
    (
      "TB(8.4e-279, 5.5e175)",
      &first,
      2.793721802689365e-175,
      0.9999999999,
    ),
    (
      "TB(1.2e-158, 1.2e192)",
      &second,
      1.3990038738849746e-191,
      0.9999999999,
    ),
    (
      "TG(1.1e-204, 6.3e-223)",
      &gamma,
      1.166432719466721e-221,
      0.9999999999,
    ),
  ]);
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

/// Nearer a finite upper bound the exponential's quantile is that bound less its offset, within an ulp of mpmath's
/// point, where the cdf returns the level to within its step across that ulp.
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
      let got = law.quantile(p).unwrap();
      assert!(
        (got - x).abs() <= x.next_up() - x,
        "{name}: quantile({p:e}) = {got:e}, mpmath {x:e}"
      );
      let step = law.pdf(x).unwrap() * (x.next_up() - x);
      let back = law.cdf(x).unwrap();
      assert!(
        (back - p).abs() <= 1e-12 * (1.0 - p) + step,
        "{name}: cdf at quantile({p:e}) = {back}"
      );
    }
  }
}

/// A lower bound inside a singular end's panel keeps the point's distance from it to 1e-12, or two ulps; mpmath's
/// references from the exponential integral, which these shapes (5e-235, 1e-170) follow far below that bar.
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

/// A shape in the hundreds at points far below 1: `ln(x/c)` from the quotient, as `ln x − ln c` would hand the shape
/// their rounding; the law is `(x/up)^a` to 1e-65 on its interval (mpmath, rounded once).
#[test]
fn a_large_shape_far_below_one() {
  let law = SimdTruncatedBeta::<f64>::new(
    9.444746241328893e2,
    4.3346600471810826e-291,
    0.0,
    2.3481719077754413e-65,
  );
  let name = "TB(944.5, 4.3e-291, [0, 2.35e-65])";
  cdfs(
    name,
    &law,
    &[
      (1.1300378217894902e-65, 1.0000000000008914e-300),
      (2.3e-65, 3.146228447239031e-9),
    ],
  );
  quantiles(
    name,
    &law,
    &[
      (1e-300, 1.1300378217894891e-65),
      (1e-10, 2.2916167667507703e-65),
    ],
  );
}

/// A normal point nearer a bound than `μ` is held beside it by `(x − bound)/σ`, and a quantile the walk from that bound
/// returns maps back from the bound itself: `(x − μ)/σ` rounds the offset away when `μ` lies far off (mpmath).
#[test]
fn a_normal_beside_a_bound_far_from_its_mean() {
  let law = SimdTruncatedNormal::<f64>::new(-0.7, 0.9, 1.3, 2.0);
  let name = "TN(−0.7, 0.9, [1.3, 2])";
  cdfs(
    name,
    &law,
    &[
      (1.30000001, 3.1844441258684016e-8),
      (1.3000000001, 3.184444447624846e-10),
    ],
  );
  quantiles(name, &law, &[(1e-10, 1.3000000000314027)]);
  let law = SimdTruncatedNormal::<f64>::new(
    -2.2781875788037683e-161,
    2.8509567226098224e-162,
    -1.8036267976644332e-162,
    2.5336181655655235e-31,
  );
  let (lo, want) = (-1.8036267976644332e-162_f64, -1.8036267976263678e-162_f64);
  let got = law.quantile(1e-10).unwrap();
  // Its offset from the bound, 3.8e-173, to 1e-12 or an ulp.
  let bar = 1e-12 * (want - lo) + (want.next_down() - want).abs();
  close(
    "TN(−2.3e-161, 2.9e-162, [−1.8e-162, 2.5e-31])",
    "quantile(1e-10)",
    got,
    want,
    0.0,
    bar,
  );
}
