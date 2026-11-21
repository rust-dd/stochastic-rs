//! A Monte Carlo price beside the closed-form Greeks: `McBarrierPricer` on an up-and-out call whose
//! barrier no path reaches, against `BsmPricer` (`cargo run --release --example mc_greeks_demo`).

use stochastic_rs::quant::OptionType;
use stochastic_rs::quant::pricing::barrier::BarrierType;
use stochastic_rs::quant::pricing::barrier::McBarrierPricer;
use stochastic_rs::quant::pricing::bsm::BsmCoc;
use stochastic_rs::quant::pricing::bsm::BsmPricer;
use stochastic_rs::quant::pricing::exotic::BarrierOption;
use stochastic_rs::quant::traits::ModelPricer;

fn main() {
  let s: f64 = 100.0;
  let k: f64 = 100.0;
  let r: f64 = 0.05;
  let sigma: f64 = 0.20;
  let t: f64 = 1.0;

  println!("=== stochastic-rs MC Greeks demo ===");
  println!("Setup: S₀={s}, K={k}, r={r:.2}, σ={sigma:.2}, T={t}, n_paths=50 000, n_steps=252\n");

  let far_barrier = BarrierOption {
    strike: k,
    barrier: 10.0 * s,
    rebate: 0.0,
    barrier_type: BarrierType::UpAndOut,
    option_type: OptionType::Call,
  };
  let mc_est = McBarrierPricer::new(sigma, 50_000, 252).price(&far_barrier, s, r, 0.0, t);

  let bs = BsmPricer::new(sigma, BsmCoc::Bsm1973);
  let bs_price = bs.price_call(s, k, r, 0.0, t);
  let greeks = bs.greeks(s, k, r, 0.0, t, OptionType::Call);

  println!(
    "Price:  BS analytic = {bs_price:.4}    MC = {:.4} ± {:.4}    rel-err = {:.4}%",
    mc_est.mean,
    mc_est.std_err,
    100.0 * (mc_est.mean - bs_price).abs() / bs_price
  );

  println!("\nGreeks aggregate (closed-form, from BsmPricer):");
  println!(
    "  Δ      = {:+.4}      Γ     = {:+.4}      V    = {:+.4}",
    greeks.delta, greeks.gamma, greeks.vega
  );
  println!(
    "  Θ      = {:+.4}      ρ     = {:+.4}",
    greeks.theta, greeks.rho
  );
  println!(
    "  vanna  = {:+.4}      charm = {:+.4}      volga= {:+.4}      veta = {:+.4}",
    greeks.vanna, greeks.charm, greeks.volga, greeks.veta
  );

  let arr = greeks.as_array();
  println!(
    "\nGreeks::as_array (length = {}, COMPONENT_NAMES order):",
    arr.len()
  );
  for (name, value) in stochastic_rs::quant::traits::Greeks::COMPONENT_NAMES
    .iter()
    .zip(arr.iter())
  {
    println!("  {name:<6} = {value:+.4}");
  }
}
