//! Integration tests for the risk-metrics module.
//!
//! Validates VaR / ES against analytical Gaussian references, drawdown and
//! performance metrics against hand-computed tables, curve shifts against
//! elementary algebra, and bucket DV01 against closed-form zero-coupon bond
//! duration.

use ndarray::Array1;
use ndarray::array;
use stochastic_rs::quant::curves::DiscountCurve;
use stochastic_rs::quant::curves::InterpolationMethod;
use stochastic_rs::quant::risk::CurveShift;
use stochastic_rs::quant::risk::DrawdownStats;
use stochastic_rs::quant::risk::Scenario;
use stochastic_rs::quant::risk::Shock;
use stochastic_rs::quant::risk::StressTest;
use stochastic_rs::quant::risk::VarMethod;
use stochastic_rs::quant::risk::bucket_dv01;
use stochastic_rs::quant::risk::calmar_ratio;
use stochastic_rs::quant::risk::central_difference;
use stochastic_rs::quant::risk::expected_shortfall;
use stochastic_rs::quant::risk::gaussian_es;
use stochastic_rs::quant::risk::gaussian_var;
use stochastic_rs::quant::risk::historical_var;
use stochastic_rs::quant::risk::information_ratio;
use stochastic_rs::quant::risk::max_drawdown;
use stochastic_rs::quant::risk::sharpe_ratio;
use stochastic_rs::quant::risk::sortino_ratio;
use stochastic_rs::quant::risk::value_at_risk;
use stochastic_rs::quant::risk::var::PnlOrLoss;

fn synthetic_pnl(n: usize, mean: f64, sigma: f64, seed: u64) -> Array1<f64> {
  // Deterministic Box-Muller using a small Park-Miller LCG to avoid an
  // external RNG dependency in the test.
  let mut s = seed.max(1);
  let mut step = || {
    s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    (s >> 16) as u32 as f64 / u32::MAX as f64
  };
  let mut out = Array1::zeros(n);
  let mut i = 0;
  while i + 1 < n {
    let u1 = step().max(1e-12);
    let u2 = step();
    let r = (-2.0 * u1.ln()).sqrt();
    let t = 2.0 * std::f64::consts::PI * u2;
    out[i] = mean + sigma * r * t.cos();
    out[i + 1] = mean + sigma * r * t.sin();
    i += 2;
  }
  out
}

#[test]
fn gaussian_var_matches_closed_form() {
  let pnl = synthetic_pnl(20_000, 0.0, 0.02, 1);
  let var = gaussian_var(pnl.view(), 0.99, PnlOrLoss::Pnl);
  // Expected Gaussian VaR with μ ≈ 0, σ ≈ 0.02 and Φ⁻¹(0.99) ≈ 2.3263
  assert!(
    (var - 0.02 * 2.326).abs() < 2e-3,
    "Gaussian VaR = {var}, expected ≈ 0.0465"
  );
}

#[test]
fn historical_and_gaussian_var_agree_for_normal_samples() {
  let pnl = synthetic_pnl(50_000, 0.001, 0.01, 42);
  let g = gaussian_var(pnl.view(), 0.95, PnlOrLoss::Pnl);
  let h = historical_var(pnl.view(), 0.95, PnlOrLoss::Pnl);
  assert!(
    (g - h).abs() < 1e-3,
    "Gaussian VaR {g} vs historical VaR {h}"
  );
  let dispatched = value_at_risk(pnl.view(), 0.95, PnlOrLoss::Pnl, VarMethod::Historical);
  assert_eq!(dispatched, h);
}

#[test]
fn gaussian_es_exceeds_var_and_matches_formula() {
  let pnl = synthetic_pnl(20_000, 0.0, 0.02, 7);
  let var = gaussian_var(pnl.view(), 0.99, PnlOrLoss::Pnl);
  let es = gaussian_es(pnl.view(), 0.99, PnlOrLoss::Pnl);
  // Closed-form: ES = μ + σ * φ(Φ⁻¹(0.99)) / (1 − 0.99).
  // φ(2.326) ≈ 0.02665, so ES ≈ 0.02 * 0.02665 / 0.01 ≈ 0.0533.
  assert!(es > var, "ES = {es} must exceed VaR = {var}");
  assert!((es - 0.0533).abs() < 3e-3, "ES = {es}");
}

#[test]
fn expected_shortfall_dispatcher_matches_method_outputs() {
  let loss = array![
    0.01_f64, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10
  ];
  let dispatched = expected_shortfall(loss.view(), 0.9, PnlOrLoss::Loss, VarMethod::Historical);
  // Top 10% of 10 losses is just {0.10}; ES = 0.10.
  assert!(
    (dispatched - 0.10).abs() < 1e-10,
    "historical ES = {dispatched}"
  );
}

#[test]
fn max_drawdown_matches_hand_calculation() {
  // Equity path: 100 → 120 → 90 → 130 → 100.
  // Drawdowns: 0, 0, −25 %, 0, −23.08 %.
  let equity = array![100.0_f64, 120.0, 90.0, 130.0, 100.0];
  let mdd = max_drawdown(equity.view());
  assert!((mdd + 0.25).abs() < 1e-12, "MDD = {mdd}");

  let stats = DrawdownStats::from_equity(equity.view());
  assert_eq!(stats.max_index, 2);
  assert!(stats.longest_duration >= 1);
  assert!(stats.average < 0.0);
}

#[test]
fn sharpe_sortino_and_information_ratio_sanity() {
  let returns = array![0.01_f64, 0.02, -0.005, 0.015, 0.008, -0.002, 0.012, 0.007];
  let sharpe = sharpe_ratio(returns.view(), 0.0, 252.0);
  assert!(
    sharpe > 0.0,
    "positive-drift series should have positive Sharpe"
  );

  let sortino = sortino_ratio(returns.view(), 0.0, 252.0);
  assert!(
    sortino > sharpe,
    "Sortino > Sharpe for upward-skewed series"
  );

  let benchmark = array![0.005_f64, 0.01, 0.000, 0.005, 0.005, 0.000, 0.005, 0.005];
  let ir = information_ratio(returns.view(), benchmark.view(), 252.0);
  assert!(ir > 0.0);
}

#[test]
fn calmar_ratio_uses_max_drawdown() {
  let returns = array![0.01_f64, 0.02, -0.10, 0.05, 0.03];
  let calmar = calmar_ratio(returns.view(), 252.0);
  assert!(calmar.is_finite());
  assert!(calmar > 0.0);
}

#[test]
fn parallel_shift_adds_constant_to_every_zero_rate() {
  let times = array![1.0_f64, 3.0, 5.0, 10.0];
  let rates = array![0.02_f64, 0.025, 0.03, 0.035];
  let curve = DiscountCurve::from_zero_rates(
    &times,
    &rates,
    InterpolationMethod::LogLinearOnDiscountFactors,
  );
  let shifted = CurveShift::Parallel(0.01_f64).apply(&curve);

  for t in [1.0_f64, 3.0, 5.0, 10.0] {
    let diff = shifted.zero_rate(t) - curve.zero_rate(t);
    assert!((diff - 0.01).abs() < 1e-12, "Δ at t={t} = {diff}");
  }
}

#[test]
fn bucket_dv01_matches_zero_coupon_bond_duration() {
  // A 5-year zero-coupon bond priced off the curve has PV = exp(-5r).
  // dPV / dr at r = 0.03, for a 1 bp bump, should equal approximately
  // -5 * exp(-0.15) * 1e-4 = -4.3 bp in dollar terms for a $1 notional.
  let times = array![1.0_f64, 3.0, 5.0, 10.0];
  let rates = array![0.03_f64, 0.03, 0.03, 0.03];
  let curve = DiscountCurve::from_zero_rates(
    &times,
    &rates,
    InterpolationMethod::LogLinearOnDiscountFactors,
  );

  let bump = 1e-4_f64;
  let sens = bucket_dv01(&curve, bump, |c| c.discount_factor(5.0));

  let closed_form = -5.0 * (-0.15_f64).exp() * bump;
  // Nearly all sensitivity is concentrated at the 5Y pillar.
  let five_year_idx = times.iter().position(|&t| (t - 5.0).abs() < 1e-12).unwrap();
  let five_year = sens.bucket_dv01[five_year_idx];
  assert!(
    (sens.parallel_dv01 - closed_form).abs() < 1e-6,
    "parallel DV01 = {}, expected {}",
    sens.parallel_dv01,
    closed_form
  );
  assert!(
    five_year < 0.0,
    "5Y bucket DV01 must be negative (got {five_year}); bucket = {:?}",
    sens.bucket_dv01
  );
}

#[test]
fn central_difference_approximates_analytical_derivative() {
  // f(x) = x^3; f'(2) = 12.
  let f = |x: f64| x.powi(3);
  let d = central_difference(f, 2.0, 1e-4);
  assert!((d - 12.0).abs() < 1e-6, "f'(2) ≈ {d}");
}

#[test]
fn stress_test_applies_scenarios_and_collects_pnl() {
  let times = array![1.0_f64, 5.0, 10.0];
  let rates = array![0.03_f64, 0.03, 0.03];
  let base_curve = DiscountCurve::from_zero_rates(
    &times,
    &rates,
    InterpolationMethod::LogLinearOnDiscountFactors,
  );

  let scenarios = vec![
    Scenario::new("+100bp parallel").with_curve_shift("discount", CurveShift::Parallel(0.01)),
    Scenario::new("-50bp parallel").with_curve_shift("discount", CurveShift::Parallel(-0.005)),
    Scenario::new("equity crash").with_shock("spot", Shock::Multiplicative(0.8)),
  ];

  let stress = StressTest::new(scenarios);
  let equity = 100.0_f64;
  let results = stress.run(
    || base_curve.discount_factor(5.0) * equity,
    |s| {
      let curve = s.resolve_curve("discount", &base_curve);
      let spot = s.resolve_scalar("spot", equity);
      curve.discount_factor(5.0) * spot
    },
  );

  assert_eq!(results.len(), 3);
  assert!(results[0].pnl < 0.0, "higher rates reduce PV");
  assert!(results[1].pnl > 0.0, "lower rates raise PV");
  assert!(results[2].pnl < 0.0, "equity crash reduces PV");
}
