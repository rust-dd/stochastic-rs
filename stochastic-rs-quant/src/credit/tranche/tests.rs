//! Loss-distribution and tranche identities: binomial limit at zero
//! correlation, all-or-nothing at full correlation, the Vasicek large-pool
//! limit, whole-capital-structure conservation and spread monotonicity.

use ndarray::Array1;
use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::special::norm_cdf;

use super::*;
use crate::credit::survival_curve::HazardInterpolation;
use crate::curves::InterpolationMethod;

fn flat_survival(h: f64) -> SurvivalCurve<f64> {
  SurvivalCurve::from_hazard_rates(
    &Array1::from_vec(vec![1.0, 5.0, 10.0]),
    &Array1::from_vec(vec![h; 3]),
    HazardInterpolation::PiecewiseConstantHazard,
  )
}

fn homogeneous_pool(n: usize, hazard: f64, recovery: f64) -> Vec<PoolName> {
  (0..n)
    .map(|_| PoolName {
      weight: 1.0 / n as f64,
      recovery,
      survival: flat_survival(hazard),
    })
    .collect()
}

fn discount() -> DiscountCurve<f64> {
  DiscountCurve::from_zero_rates(
    &Array1::from_vec(vec![0.5, 1.0, 5.0, 10.0]),
    &Array1::from_vec(vec![0.03; 4]),
    InterpolationMethod::LogLinearOnDiscountFactors,
  )
}

fn annual(years: usize) -> Vec<f64> {
  (1..=years).map(|i| i as f64).collect()
}

/// At `ρ = 0` the loss count is binomial: the recursion must reproduce it
/// exactly on a grid whose bucket equals one name's loss.
#[test]
fn zero_correlation_loss_distribution_is_binomial() {
  let n = 10;
  let recovery = 0.4;
  let pool = homogeneous_pool(n, 0.05, recovery);
  let tranche = CdoTranche::new(0.0, 1.0, 0.01, annual(5), 1.0, 0.0).with_resolution(20, n * 1000);
  let t = 3.0;
  let p_true = pool[0].survival.default_probability(t);
  // The factor integrand round-trips p through the crate's `ndtri`/`norm_cdf`
  // pair (Abramowitz–Stegun erf, 1.5e-7), so the recursion sees this p.
  let p = norm_cdf(ndtri(p_true));
  assert!((p - p_true).abs() < 1e-6);
  let dist = tranche.loss_distribution(&pool, t);
  let unit_loss = ((1.0 - recovery) / n as f64 * (n * 1000) as f64).round() as usize;
  let mut binom = 1.0;
  for k in 0..=n {
    let prob = binom * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32);
    assert!(
      (dist[k * unit_loss] - prob).abs() < 1e-12,
      "k {k}: {} vs {prob}",
      dist[k * unit_loss]
    );
    binom *= (n - k) as f64 / (k + 1) as f64;
  }
  let total: f64 = dist.iter().sum();
  assert!((total - 1.0).abs() < 1e-12);
}

/// The `[0, 100 %]` tranche carries the whole pool loss, so its expected loss
/// is the pool's expected loss whatever the correlation.
#[test]
fn whole_capital_structure_loss_equals_the_pool_expected_loss() {
  let pool = homogeneous_pool(25, 0.03, 0.4);
  let expected = 0.6 * pool[0].survival.default_probability(4.0);
  for rho in [0.0, 0.3, 0.7] {
    let whole = CdoTranche::new(0.0, 1.0, 0.0, annual(5), 1.0, rho).with_resolution(40, 500);
    let etl = whole.expected_tranche_loss(&pool, 4.0);
    assert!(
      (etl - expected).abs() < 2e-3 * expected.max(1e-3),
      "rho {rho}: {etl} vs {expected}"
    );
  }
}

/// Correlation moves risk from the equity to the senior tranche.
#[test]
fn equity_spread_falls_and_senior_spread_rises_with_correlation() {
  let pool = homogeneous_pool(50, 0.02, 0.4);
  let d = discount();
  let spread = |a: f64, dd: f64, rho: f64| {
    CdoTranche::new(a, dd, 0.0, annual(5), 1.0, rho)
      .valuation(&pool, &d)
      .fair_spread
  };
  let (equity_lo, equity_hi) = (spread(0.0, 0.03, 0.1), spread(0.0, 0.03, 0.6));
  let (senior_lo, senior_hi) = (spread(0.12, 0.22, 0.1), spread(0.12, 0.22, 0.6));
  assert!(equity_lo > equity_hi, "equity {equity_lo} -> {equity_hi}");
  assert!(senior_lo < senior_hi, "senior {senior_lo} -> {senior_hi}");
  assert!(equity_hi > senior_hi);
}

/// A 250-name homogeneous pool sits close to the Vasicek large-pool limit.
#[test]
fn large_pool_matches_the_vasicek_limit() {
  let pool = homogeneous_pool(250, 0.02, 0.4);
  let tranche = CdoTranche::new(0.03, 0.07, 0.0, annual(5), 1.0, 0.3).with_resolution(60, 1000);
  let p = pool[0].survival.default_probability(5.0);
  let recursion = tranche.expected_tranche_loss(&pool, 5.0);
  let lhp = tranche.large_pool_expected_tranche_loss(p, 0.6);
  assert!(
    (recursion - lhp).abs() < 0.02 * lhp.max(1e-4),
    "recursion {recursion} vs lhp {lhp}"
  );
}

/// Near-full correlation makes the pool default as one. The factor integrand
/// is then almost a step in `Y`, so the Gauss–Hermite rule needs many nodes;
/// the no-loss mass is checked against a fine Simpson integration of the
/// same conditional probability, and the mass must sit at zero and at the
/// full loss-given-default.
#[test]
fn full_correlation_is_all_or_nothing() {
  let pool = homogeneous_pool(20, 0.04, 0.4);
  let rho = 0.999_f64;
  let tranche = CdoTranche::new(0.0, 1.0, 0.0, annual(5), 1.0, rho).with_resolution(200, 200);
  let dist = tranche.loss_distribution(&pool, 5.0);
  let p = pool[0].survival.default_probability(5.0);
  let c = ndtri(p);
  // Simpson on [−10, 10] of (1 − p(y))²⁰ φ(y): the exact no-loss mass at this ρ.
  let panels = 40_000;
  let h = 20.0 / panels as f64;
  let mut exact = 0.0;
  for i in 0..=panels {
    let y = -10.0 + i as f64 * h;
    let survive = 1.0 - norm_cdf((c - rho.sqrt() * y) / (1.0 - rho).sqrt());
    let weight = if i == 0 || i == panels {
      1.0
    } else if i % 2 == 1 {
      4.0
    } else {
      2.0
    };
    exact += weight * survive.powi(20) * (-0.5 * y * y).exp() / (2.0 * std::f64::consts::PI).sqrt();
  }
  exact *= h / 3.0;
  let full = (0.6_f64 * 200.0).round() as usize;
  assert!(
    (dist[0] - exact).abs() < 0.02,
    "no-loss mass {} vs exact {exact}",
    dist[0]
  );
  assert!(
    (exact - (1.0 - p)).abs() < 0.05,
    "at ρ = 0.999 the exact no-loss mass approaches 1 − p = {}",
    1.0 - p
  );
  assert!(
    dist[0] + dist[full] > 0.9,
    "mass concentrates at no loss and full loss"
  );
}

#[test]
fn valuation_pieces_are_consistent() {
  let pool = homogeneous_pool(30, 0.03, 0.4);
  let tranche = CdoTranche::new(0.03, 0.06, 0.05, annual(5), 1.0, 0.25);
  let v = tranche.valuation(&pool, &discount());
  assert_eq!(v.expected_loss.len(), 5);
  assert!(v.expected_loss.windows(2).all(|w| w[1] >= w[0]));
  assert!((v.premium_leg - 0.05 * v.risky_annuity).abs() < 1e-15);
  assert!((v.upfront - (v.protection_leg - v.premium_leg)).abs() < 1e-15);
  assert!((v.fair_spread * v.risky_annuity - v.protection_leg).abs() < 1e-15);
}

#[test]
#[should_panic(expected = "need 0 ≤ A < D ≤ 1")]
fn rejects_an_inverted_tranche() {
  let _ = CdoTranche::new(0.1, 0.05, 0.0, annual(5), 1.0, 0.2);
}
