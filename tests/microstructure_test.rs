//! Integration tests for the `quant::microstructure` module.

use ndarray::Array1;
use ndarray::array;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::quant::microstructure::AlmgrenChrissParams;
use stochastic_rs::quant::microstructure::ExecutionDirection;
use stochastic_rs::quant::microstructure::ImpactKernel;
use stochastic_rs::quant::microstructure::corwin_schultz_spread;
use stochastic_rs::quant::microstructure::effective_spread;
use stochastic_rs::quant::microstructure::multi_period_kyle;
use stochastic_rs::quant::microstructure::optimal_execution;
use stochastic_rs::quant::microstructure::propagator_impact_path;
use stochastic_rs::quant::microstructure::propagator_price_impact;
use stochastic_rs::quant::microstructure::roll_spread;
use stochastic_rs::quant::microstructure::single_period_kyle;

#[test]
fn ac_efficient_frontier_monotonic_in_lambda() {
  let lambdas = [0.0_f64, 0.1, 1.0, 10.0, 100.0];
  let mut last_cost = -f64::INFINITY;
  let mut last_var = f64::INFINITY;
  for &lam in &lambdas {
    let p = AlmgrenChrissParams::new(10_000.0, 1.0, 50, 0.02, 1e-7, 1e-5, lam);
    let plan = optimal_execution(&p);
    assert!(plan.expected_cost >= last_cost - 1e-9);
    assert!(plan.variance <= last_var + 1e-9);
    last_cost = plan.expected_cost;
    last_var = plan.variance;
  }
}

#[test]
fn ac_buy_and_sell_are_mirror_images() {
  // After the rc.2 fix to Almgren-Chriss `Buy` direction, all three series
  // (`inventory`, `trades`, `rates`) flip sign so consumers see consistent
  // buy-frame numbers. Previously only `rates` was flipped, leaving
  // `inventory` and `trades` in the sell frame — see audit §1.4.7.
  let mut p = AlmgrenChrissParams::new(2_000.0_f64, 1.0, 25, 0.025, 5e-8, 5e-6, 1.0);
  let sell = optimal_execution(&p);
  p.direction = ExecutionDirection::Buy;
  let buy = optimal_execution(&p);
  for i in 0..p.n_intervals {
    assert!(
      (sell.trades[i] + buy.trades[i]).abs() < 1e-12,
      "trades mirror: sell={}, buy={}",
      sell.trades[i],
      buy.trades[i]
    );
    assert!(
      (sell.rates[i] + buy.rates[i]).abs() < 1e-12,
      "rates mirror: sell={}, buy={}",
      sell.rates[i],
      buy.rates[i]
    );
  }
  for i in 0..=p.n_intervals {
    assert!(
      (sell.inventory[i] + buy.inventory[i]).abs() < 1e-12,
      "inventory mirror: sell={}, buy={}",
      sell.inventory[i],
      buy.inventory[i]
    );
  }
}

#[test]
fn kyle_single_period_satisfies_constants() {
  let eq = single_period_kyle(0.16_f64, 0.25);
  assert!((eq.beta * eq.lambda - 0.5).abs() < 1e-12);
  assert!((2.0 * eq.posterior_variance - 0.16).abs() < 1e-12);
}

#[test]
fn kyle_multi_period_terminal_satisfies_static_kyle_product() {
  // Canonical Kyle (1985) / Cetin-Larsen 2023 Thm 2.1: at the terminal round
  // γ_N = α_N λ_N = 0 ⇒ β_N λ_N = (1 − 2γ_N) / (2(1 − γ_N)) = 1/2. (rc.0
  // shipped a non-canonical recursion that gave 1/4 at the terminal — that
  // bug was fixed in rc.1 by re-deriving against Cetin-Larsen 2023.)
  let eqs = multi_period_kyle(1.0_f64, 1.0, 6);
  let last = eqs.last().unwrap();
  assert!((last.beta * last.lambda - 0.5).abs() < 1e-9);
}

#[test]
fn impact_propagator_path_is_cumulative() {
  let v = array![1.0_f64, 1.0, -1.0, -1.0, 1.0];
  let path = propagator_impact_path(v.view(), ImpactKernel::PowerLaw, 0.3, 0.6);
  for t in 0..v.len() {
    let p = propagator_price_impact(v.slice(ndarray::s![..=t]), ImpactKernel::PowerLaw, 0.3, 0.6);
    assert!((path[t] - p).abs() < 1e-12);
  }
}

#[test]
fn roll_recovers_spread_under_simulated_bounce() {
  let mid = 50.0_f64;
  let s = 0.05;
  let dist = SimdNormal::<f64>::with_seed(0.0, 1.0, 17);
  let mut signs = vec![0.0_f64; 20_000];
  dist.fill_slice_fast(&mut signs);
  let p = Array1::from_iter(signs.iter().map(|&z| {
    let sign = if z >= 0.0 { 1.0 } else { -1.0 };
    mid + 0.5 * s * sign
  }));
  let est = roll_spread(p.view());
  assert!((est - s).abs() < 0.02);
}

#[test]
fn effective_spread_zero_when_trades_at_mid() {
  let p = Array1::<f64>::from_elem(4, 100.0);
  let m = Array1::<f64>::from_elem(4, 100.0);
  assert!(effective_spread(p.view(), m.view()).abs() < 1e-12);
}

#[test]
fn corwin_schultz_increases_with_high_low_range() {
  let h_tight = array![100.1_f64, 100.2, 100.15];
  let l_tight = array![99.9_f64, 99.8, 99.85];
  let h_wide = array![101.0_f64, 102.0, 101.5];
  let l_wide = array![99.0_f64, 98.0, 98.5];
  let s_tight = corwin_schultz_spread(h_tight.view(), l_tight.view());
  let s_wide = corwin_schultz_spread(h_wide.view(), l_wide.view());
  assert!(s_wide > s_tight);
}
