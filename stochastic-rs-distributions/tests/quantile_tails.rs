//! Quantiles at the far ends of their levels and parameters against mpmath (50 and 90 digits agreeing).

use stochastic_rs_distributions::DistributionExt;
use stochastic_rs_distributions::cauchy::SimdCauchy;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::ged::SimdGed;
use stochastic_rs_distributions::geometric::SimdGeometric;
use stochastic_rs_distributions::pareto::SimdPareto;
use stochastic_rs_distributions::truncated::SimdTruncatedExp;

fn rel(a: f64, b: f64) -> f64 {
  (a - b).abs() / b.abs()
}

/// Where the gamma root underflows, in either branch, its `1/β` power is `level·Γ(1 + 1/β)`; β = 20 takes the root.
#[test]
fn ged_quantile_at_large_beta_keeps_its_offset() {
  for (beta, p, want) in [
    (1000.0, 0.3, -0.39976950899383823),
    (1000.0, 0.45, -0.09994237724845953),
    (1000.0, 0.49, -0.019988475449691928),
    (1000.0, 0.7, 0.3997695089938381),
    (100.0, 0.4999, -0.0001988651702382793),
    (50.0, 0.5 - 2f64.powi(-30), -1.8418658585546784e-9),
    (25.0, 0.5 + 2f64.powi(-52), 4.3451384753112375e-16),
    (20.0, 0.3, -0.3894017063442717),
    (20.0, 0.5 - 2f64.powi(-30), -1.8132929980061496e-9),
    (2000.0, 0.2, -0.5998269835909047),
    (1074.0, 1e-10, -1.0023722824375438),
    (1074.0, 0.2, -0.5996780471831865),
    (1074.0, 0.8, 0.5996780471831866),
    (1074.0, 1.0 - 1e-10, 1.002372282431897),
    (1e4, 1e-10, -1.000236789374925),
    (1e4, 0.2, -0.5999653729938974),
    (1e4, 0.8, 0.5999653729938975),
    (1e4, 1.0 - 1e-10, 1.0002367893742115),
  ] {
    let got = SimdGed::<f64>::new(0.0, 1.0, beta).quantile(p).unwrap();
    assert!(
      rel(got, want) < 1e-13,
      "β = {beta}, p = {p:e}: {got:e}, want {want:e}"
    );
  }
}

/// Above ½ `1 − p·m` as `(1 − p) + p·e^{−λL}` keeps what the map cancels, next to level 1 and far below `upper`.
#[test]
fn truncated_exponential_quantile_keeps_its_upper_tail() {
  let levels = [1.0 - 2f64.powi(-53), 1.0 - 1e-15, 1.0 - 1e-10];
  for ((lam, lo, up), wants) in [
    (
      (40.0, 1.0, 2.0),
      [1.9174812193520643, 1.8633833313228805, 1.5756462701179137],
    ),
    (
      (25.0, 0.0, 1.0),
      [0.9999996802352896, 0.9999971222096268, 0.9158322410743698],
    ),
    (
      (30.0, 0.0, 1.0),
      [0.999960475541565, 0.9996459557801568, 0.7674971840844416],
    ),
  ] {
    let law = SimdTruncatedExp::<f64>::new(lam, lo, up);
    for (p, want) in levels.into_iter().zip(wants) {
      let got = law.quantile(p).unwrap();
      assert!(
        rel(got, want) < 1e-14,
        "({lam}, {lo}, {up}) at 1 − p = {:e}: {got}, want {want}",
        1.0 - p
      );
    }
  }
  let bulk = SimdTruncatedExp::<f64>::new(7.0, 0.0, 101.39)
    .quantile(0.5180645342510322)
    .unwrap();
  assert!(rel(bulk, 0.104277866054717) < 1e-15, "{bulk}");
  assert_eq!(
    SimdTruncatedExp::<f64>::new(1e-17, 0.0, 1.0).quantile(1e-300),
    Some(1e-300)
  );
}

/// The map below ½ and the two-term form above meet without a step down, over 10⁴ floats on each side.
#[test]
fn truncated_exponential_quantile_is_monotone_through_one_half() {
  for (lam, lo, up) in [(25.0, 0.0, 1.0), (7.0, 0.0, 101.39), (0.5, 0.0, 3.0)] {
    let law = SimdTruncatedExp::<f64>::new(lam, lo, up);
    let mut p = 0.5_f64;
    for _ in 0..10_000 {
      p = p.next_down();
    }
    let mut last = law.quantile(p).unwrap();
    for _ in 0..20_000 {
      p = p.next_up();
      let q = law.quantile(p).unwrap();
      assert!(
        q >= last,
        "({lam}, {lo}, {up}) at p = {p}: {q} after {last}"
      );
      last = q;
    }
  }
}

/// Past `e^{−λL}`'s underflow the bounded moments meet `∞·0`; the open tail's values hold there.
#[test]
fn truncated_exponential_moments_at_a_huge_finite_upper() {
  for (lam, up, entropy) in [
    (2.0, f64::MAX, 0.3068528194400547),
    (10.0, 1e308, -1.3025850929940457),
    (1.0, 1e160, 1.0),
    (1e-3, 1e200, 7.907755278982137),
  ] {
    let law = SimdTruncatedExp::<f64>::new(lam, 0.0, up);
    assert!(
      rel(law.mean().unwrap(), 1.0 / lam) < 1e-15,
      "λ = {lam}: mean"
    );
    assert!(
      rel(law.variance().unwrap(), 1.0 / (lam * lam)) < 1e-15,
      "λ = {lam}: variance"
    );
    assert!(
      rel(law.entropy().unwrap(), entropy) < 1e-15,
      "λ = {lam}: entropy"
    );
  }
}

/// θ enters the solver's exponent, so a unit root below the normals keeps its digits at any scale, and a subnormal
/// result rounds once.
#[test]
fn gamma_quantile_at_a_large_scale() {
  for (shape, scale, p, want) in [
    (0.5, 1e10, 1e-158, 7.853981633974484e-307),
    (0.5, 1e30, 1e-160, 7.853981633974483e-291),
    (0.5, 1e100, 1e-200, 7.853981633974482e-301),
    (0.01, 1e300, 1e-5, 5.6607381470620634e-201),
    (
      0.5678006476703295,
      2.3367910557514248e119,
      3.477217825905003e-176,
      1.8407855932429324e-190,
    ),
  ] {
    let got = SimdGamma::<f64>::new(shape, scale).quantile(p).unwrap();
    assert!(
      rel(got, want) < 1e-15,
      "({shape}, {scale:e}) at {p:e}: {got:e}"
    );
  }
  assert_eq!(
    SimdGamma::<f64>::new(0.5, 1e30).quantile(1e-170),
    Some(7.8539816339746e-311)
  );
  assert_eq!(
    SimdGamma::<f64>::new(5.930266202879256, 9.635923662524929e-267)
      .quantile(1.7547492875562472e-273),
    Some(2.89952531937e-312)
  );
}

/// The cot's first order to half an ulp: `(γ/π)/p` at a subnormal level, where `πp` would round, and `γ/(πp)` at a
/// normal one, where a tiny `γ/π` would.
#[test]
fn cauchy_quantile_at_a_subnormal_level() {
  let got = SimdCauchy::<f64>::new(0.0, 1e-5).quantile(1e-310).unwrap();
  assert!(rel(got, -3.1830988618379165e304) < 1e-15, "{got:e}");
  let tiny = SimdCauchy::<f64>::new(0.0, 5e-324).quantile(1e-20).unwrap();
  assert!(rel(tiny, -1.5726597949504822e-304) < 1e-15, "{tiny:e}");
}

/// One form at every level keeps the quantile monotone through ½; a form switched at ½ can step down an ulp there.
#[test]
fn pareto_quantile_is_monotone_through_one_half() {
  let law = SimdPareto::<f64>::new(1.0, 7.244359600749906);
  assert!(law.quantile(0.5 - 2f64.powi(-54)).unwrap() <= law.quantile(0.5).unwrap());
}

/// At `p = 1` the median is the one support point, as every level's quantile is.
#[test]
fn geometric_median_at_p_one_is_the_support_point() {
  assert_eq!(SimdGeometric::<u64>::new(1.0).median(), Some(1.0));
}
