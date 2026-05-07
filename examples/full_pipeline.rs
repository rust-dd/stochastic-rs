//! End-to-end pricing pipeline demonstrating the v2 architecture.
//!
//! ```sh
//! cargo run --release --example full_pipeline
//! ```
//!
//! Walks through:
//!   1. Build reactive market handles (`SimpleQuote` + `Handle`).
//!   2. Wire two pricing engines (`AnalyticBSEngine`, `AnalyticHestonEngine`)
//!      against the same market handles.
//!   3. Define a multi-instrument portfolio: equity option, variance swap,
//!      total return swap.
//!   4. Show reactive re-pricing — change a spot quote and watch every
//!      engine output update.
//!   5. Build an FX vol smile (Vanna–Volga from ATM/RR/BFLY quotes).
//!   6. Aggregate Greeks and run parametric VaR on simulated PnL.

use std::sync::Arc;

use ndarray::ArrayView1;
use stochastic_rs::quant::OptionType;
use stochastic_rs::quant::fx::AtmConvention;
use stochastic_rs::quant::fx::FxDeltaConvention;
use stochastic_rs::quant::fx::FxMarketQuotes;
use stochastic_rs::quant::fx::VannaVolgaSmile;
use stochastic_rs::quant::instruments::EuropeanOption;
use stochastic_rs::quant::instruments::TotalReturnSwap;
use stochastic_rs::quant::instruments::TrsDirection;
use stochastic_rs::quant::instruments::TrsPeriod;
use stochastic_rs::quant::market::Handle;
use stochastic_rs::quant::market::Quote;
use stochastic_rs::quant::market::SimpleQuote;
use stochastic_rs::quant::microstructure::AlmgrenChrissParams;
use stochastic_rs::quant::microstructure::ExecutionDirection;
use stochastic_rs::quant::pricing::AnalyticBSEngine;
use stochastic_rs::quant::pricing::AnalyticHestonEngine;
use stochastic_rs::quant::pricing::HestonStaticParams;
use stochastic_rs::quant::pricing::VarianceSwapPricer;
use stochastic_rs::quant::pricing::execution_adjusted_price;
use stochastic_rs::quant::risk::liquidity_adjusted_var;
use stochastic_rs::quant::risk::var::PnlOrLoss;
use stochastic_rs::quant::risk::var::VarMethod;
use stochastic_rs::quant::risk::var::gaussian_var;
use stochastic_rs::traits::PricingEngine;
use stochastic_rs::traits::PricingResult;

fn main() {
  println!("=========================================================");
  println!("  stochastic-rs — full pricing pipeline demo (v2)");
  println!("=========================================================\n");

  // -----------------------------------------------------------------------
  // 1. Reactive market handles
  // -----------------------------------------------------------------------
  let spot_q = Arc::new(SimpleQuote::new(100.0));
  let vol_q = Arc::new(SimpleQuote::new(0.20));
  let rate_q = Arc::new(SimpleQuote::new(0.05));
  let div_q = Arc::new(SimpleQuote::new(0.02));

  let spot = Handle::new(Arc::clone(&spot_q));
  let vol = Handle::new(Arc::clone(&vol_q));
  let rate = Handle::new(Arc::clone(&rate_q));
  let div = Handle::new(Arc::clone(&div_q));

  println!("Market handles:");
  println!("  spot   = {:.4}", spot_q.value());
  println!("  vol    = {:.4}", vol_q.value());
  println!("  rate   = {:.4}", rate_q.value());
  println!("  div    = {:.4}\n", div_q.value());

  // -----------------------------------------------------------------------
  // 2. Two engines against the same market
  // -----------------------------------------------------------------------
  let bs_engine = AnalyticBSEngine::new(spot.clone(), vol.clone(), rate.clone(), div.clone());

  let heston_params = HestonStaticParams::new(0.04, 1.5, 0.04, 0.30, -0.7);
  let heston_engine = AnalyticHestonEngine::new(spot.clone(), rate.clone(), div.clone(), heston_params);

  // -----------------------------------------------------------------------
  // 3. Portfolio: equity option + variance swap + TRS
  // -----------------------------------------------------------------------
  let call = EuropeanOption::new_tau(110.0, OptionType::Call, 1.0);
  let put = EuropeanOption::new_tau(95.0, OptionType::Put, 1.0);

  let bs_call = bs_engine.calculate(&call);
  let heston_call = heston_engine.calculate(&call);
  let bs_put = bs_engine.calculate(&put);

  println!("Portfolio leg 1 — equity options (Black-Scholes vs Heston):");
  println!("  Call(K=110, T=1) BS    : NPV={:.4}  Δ={:.4}", bs_call.npv(), bs_call.greeks().unwrap().delta);
  println!("  Call(K=110, T=1) Heston: NPV={:.4}  Δ={:.4}", heston_call.npv(), heston_call.greeks().unwrap().delta);
  println!("  Put (K= 95, T=1) BS    : NPV={:.4}  Δ={:.4}\n", bs_put.npv(), bs_put.greeks().unwrap().delta);

  let var_swap = VarianceSwapPricer {
    s: spot_q.value(),
    r: rate_q.value(),
    q: div_q.value(),
    t: 1.0,
  };
  let k_var_bsm = var_swap.fair_strike_bsm(vol_q.value());
  let k_var_heston = var_swap.fair_strike_heston(
    heston_params.v0,
    heston_params.kappa,
    heston_params.theta,
  );

  println!("Portfolio leg 2 — variance swap (fair variance strike):");
  println!("  K_var (BSM,  σ=20%) = {:.6}  → vol = {:.4}%", k_var_bsm, k_var_bsm.sqrt() * 100.0);
  println!("  K_var (Heston BL00) = {:.6}  → vol = {:.4}%\n", k_var_heston, k_var_heston.sqrt() * 100.0);

  let trs = TotalReturnSwap {
    notional: 10_000_000.0,
    spot: spot_q.value(),
    equity_drift_rate: rate_q.value(),
    schedule: (1..=4)
      .map(|i| TrsPeriod {
        end_time: i as f64 * 0.25,
        accrual: 0.25,
        funding_rate: rate_q.value(),
      })
      .collect(),
    spread: 0.0,
    direction: TrsDirection::ReceiveEquity,
  };
  let trs_v = trs.value_flat(rate_q.value());
  println!("Portfolio leg 3 — total return swap (1y, quarterly resets):");
  println!("  equity-leg PV  = {:>14.2}", trs_v.equity_leg_pv);
  println!("  funding-leg PV = {:>14.2}", trs_v.funding_leg_pv);
  println!("  fair spread    = {:.6} ({:.2} bps)\n", trs_v.fair_spread, trs_v.fair_spread * 1e4);

  // -----------------------------------------------------------------------
  // 4. Reactive update: bump spot, all engines re-price automatically
  // -----------------------------------------------------------------------
  println!("Reactive update — spot 100 → 105:");
  spot_q.set_value(105.0);
  let bs_after = bs_engine.calculate(&call);
  let heston_after = heston_engine.calculate(&call);
  println!("  Call BS     NPV: {:.4} → {:.4}  (Δ ~ {:.4})", bs_call.npv(), bs_after.npv(), bs_after.greeks().unwrap().delta);
  println!("  Call Heston NPV: {:.4} → {:.4}\n", heston_call.npv(), heston_after.npv());
  spot_q.set_value(100.0);

  // -----------------------------------------------------------------------
  // 5. FX vol smile (Vanna-Volga)
  // -----------------------------------------------------------------------
  let fx_quotes = FxMarketQuotes {
    atm: 0.10,
    rr_25: -0.005,
    bf_25: 0.0015,
    atm_convention: AtmConvention::DeltaNeutralStraddle,
    delta_convention: FxDeltaConvention::Forward,
  };
  let fwd = 1.10;
  let smile = VannaVolgaSmile::build(fx_quotes, fwd, 0.5, 0.02);
  println!("FX vol smile (EUR/USD, 6m, Vanna-Volga):");
  println!("  pivots: K_25P={:.4}  K_ATM={:.4}  K_25C={:.4}", smile.k_put, smile.k_atm, smile.k_call);
  for k_step in [-0.05, -0.025, 0.0, 0.025, 0.05] {
    let k = fwd + k_step;
    let v = smile.vol_at_strike(k);
    println!("  K={:.4} → σ={:.4} ({:.2}%)", k, v, v * 100.0);
  }
  println!();

  // -----------------------------------------------------------------------
  // 6. Aggregate Greeks
  // -----------------------------------------------------------------------
  let g_call = bs_call.greeks().unwrap();
  let g_put = bs_put.greeks().unwrap();
  let net_delta = g_call.delta - g_put.delta; // long call, short put
  let net_gamma = g_call.gamma + g_put.gamma;
  let net_vega = g_call.vega + g_put.vega;
  println!("Aggregate Greeks (long call − short put):");
  println!("  Δ = {:.4}", net_delta);
  println!("  Γ = {:.6}", net_gamma);
  println!("  ν = {:.4}\n", net_vega);

  // -----------------------------------------------------------------------
  // 7. Parametric VaR on simulated portfolio PnL
  // -----------------------------------------------------------------------
  let n_sim = 1_000_usize;
  let pnl: Vec<f64> = (0..n_sim)
    .map(|i| {
      let z = (i as f64 - n_sim as f64 / 2.0) / (n_sim as f64 / 6.0);
      net_delta * 100.0 * 0.01 * z + 0.5 * net_gamma * (100.0_f64 * 0.01 * z).powi(2)
    })
    .collect();
  let pnl_arr = ndarray::Array1::from(pnl);
  let var_95 = gaussian_var(ArrayView1::from(&pnl_arr), 0.95, PnlOrLoss::Pnl);
  let var_99 = gaussian_var(ArrayView1::from(&pnl_arr), 0.99, PnlOrLoss::Pnl);
  println!("Portfolio risk (parametric 1d Gaussian, n={}):", n_sim);
  println!("  VaR(95%) = {:.4}", var_95);
  println!("  VaR(99%) = {:.4}\n", var_99);

  // -----------------------------------------------------------------------
  // 8. Liquidity-adjusted view (Almgren-Chriss execution cost)
  // -----------------------------------------------------------------------
  let exec = AlmgrenChrissParams {
    total_shares: 50_000.0,
    horizon: 1.0,
    n_intervals: 20,
    direction: ExecutionDirection::Sell,
    volatility: vol_q.value(),
    gamma: 1e-7,
    eta: 5e-6,
    epsilon: 0.0,
    lambda: 0.5,
  };
  let liquidity_var_95 = liquidity_adjusted_var(
    ArrayView1::from(&pnl_arr),
    0.95,
    PnlOrLoss::Pnl,
    VarMethod::Historical,
    &exec,
  );
  let exec_adj_call = execution_adjusted_price(bs_call.npv(), &exec, exec.lambda);
  println!("Liquidity-adjusted view (Almgren-Chriss, 50k shares, T=1d):");
  println!("  VaR(95%) raw     = {:.4}", var_95);
  println!("  VaR(95%) +exec   = {:.4}", liquidity_var_95);
  println!("  BS call NPV raw  = {:.4}", bs_call.npv());
  println!("  BS call +exec    = {:.4}\n", exec_adj_call);

  println!("=========================================================");
  println!("  All blocks executed against the v2 trait surface.");
  println!("=========================================================");
}
