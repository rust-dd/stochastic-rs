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

/// Where the gamma root underflows, its `1/β` power is `level·Γ(1 + 1/β)`, still representable; β = 20 takes the root.
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
  ] {
    let got = SimdGed::<f64>::new(0.0, 1.0, beta).quantile(p).unwrap();
    assert!(
      rel(got, want) < 1e-13,
      "β = {beta}, p = {p:e}: {got:e}, want {want:e}"
    );
  }
}

/// Above ½ the map referred to `upper` keeps the `e^{−λL}` that `1 − p·m` cancels next to level 1.
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
  assert_eq!(
    SimdTruncatedExp::<f64>::new(1e-17, 0.0, 1.0).quantile(1e-300),
    Some(1e-300)
  );
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

/// θ's power of two enters the solver, so a unit root below the normals keeps its digits at a large scale.
#[test]
fn gamma_quantile_at_a_large_scale() {
  let low = SimdGamma::<f64>::new(0.5, 1e10).quantile(1e-158).unwrap();
  assert!(rel(low, 7.853981633974484e-307) < 1e-13, "{low:e}");
  let lower = SimdGamma::<f64>::new(0.5, 1e30).quantile(1e-160).unwrap();
  assert!(rel(lower, 7.853981633974483e-291) < 1e-13, "{lower:e}");
}

/// At a subnormal level `πp` would round; `(γ/π)/p` is the cot's first order to half an ulp there.
#[test]
fn cauchy_quantile_at_a_subnormal_level() {
  let got = SimdCauchy::<f64>::new(0.0, 1e-5).quantile(1e-310).unwrap();
  assert!(rel(got, -3.1830988618379165e304) < 1e-15, "{got:e}");
}

/// One form at every level keeps the quantile monotone through ½, where a split at ½ stepped down an ulp.
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
