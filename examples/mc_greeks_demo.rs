//! End-to-end Monte Carlo Greeks demo.
//!
//! Prices a vanilla European call via `MCBarrierPricer` (with a far-OTM
//! `UpAndOut` barrier so no path ever hits it — gives a plain vanilla
//! payoff) and reports the full first- and second-order Greek aggregate
//! (the analytic `BSMPricer`'s closed-form Greeks, collected into a
//! `Greeks` aggregate).
//!
//! The MC pricer is wired in as the *price* engine; the Greeks come from
//! the canonical analytic reference because `MCBarrierPricer::price` does
//! not accept an explicit seed, so a naive central-difference over
//! independent re-runs would be dominated by sampling noise (gamma in
//! particular needs pathwise / Malliavin estimators — see
//! `stochastic_rs::quant::pricing::malliavin_thalmaier` for a worked
//! example with the proper variance-reduction wiring).
//!
//! Run with:
//!   cargo run --release --example mc_greeks_demo

use stochastic_rs::quant::OptionType;
use stochastic_rs::quant::pricing::barrier::BarrierType;
use stochastic_rs::quant::pricing::barrier::MCBarrierPricer;
use stochastic_rs::quant::pricing::bsm::BSMCoc;
use stochastic_rs::quant::pricing::bsm::BSMPricer;
use stochastic_rs::quant::traits::Greeks;
use stochastic_rs::quant::traits::ModelPricer;

fn main() {
  let s: f64 = 100.0;
  let k: f64 = 100.0;
  let r: f64 = 0.05;
  let sigma: f64 = 0.20;
  let t: f64 = 1.0;

  println!("=== stochastic-rs MC Greeks demo ===");
  println!("Setup: S₀={s}, K={k}, r={r:.2}, σ={sigma:.2}, T={t}, n_paths=50 000, n_steps=252\n");

  // 1) Monte Carlo price via MCBarrierPricer with a barrier high enough
  //    that no realistic GBM path hits it ⇒ vanilla European call payoff.
  let mc = MCBarrierPricer {
    n_paths: 50_000,
    n_steps: 252,
  };
  let huge_barrier = 10.0 * s;
  let mc_price = mc.price(
    s,
    k,
    huge_barrier,
    r,
    sigma,
    t,
    BarrierType::UpAndOut,
    OptionType::Call,
  );

  // 2) Analytic BS reference + the full first-/second-order Greek set.
  //    `BSMPricer` holds only (v, coc); the (s, k, r, q, tau) query and
  //    the option type travel to each call.
  let bs = BSMPricer::new(sigma, BSMCoc::Bsm1973);
  let ot = OptionType::Call;
  let bs_price = bs.price_call(s, k, r, 0.0, t);
  let greeks = Greeks {
    delta: bs.delta(s, k, r, 0.0, t, ot),
    gamma: bs.gamma(s, k, r, 0.0, t),
    vega: bs.vega(s, k, r, 0.0, t),
    theta: bs.theta(s, k, r, 0.0, t, ot),
    rho: bs.rho(s, k, r, 0.0, t, ot),
    vanna: bs.vanna(s, k, r, 0.0, t),
    charm: bs.charm(s, k, r, 0.0, t, ot),
    volga: bs.vomma(s, k, r, 0.0, t),
    veta: bs.dvega_dtime(s, k, r, 0.0, t),
  };

  println!(
    "Price:  BS analytic = {bs_price:.4}    MC = {mc_price:.4}    rel-err = {:.4}%",
    100.0 * (mc_price - bs_price).abs() / bs_price
  );

  // 3) Full Greeks struct from the analytic side (first- and second-order).
  //    A production MC Greeks pipeline should plug into the Malliavin
  //    family (see `pricing::malliavin_thalmaier`) rather than a naive
  //    central-difference re-run, because the latter is dominated by
  //    sampling noise on Γ, Θ, vanna, charm, volga, veta.
  println!("\nGreeks aggregate (closed-form, from BSMPricer):");
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

  // 4) Greeks::as_array → 9-element canonical layout, stable positional
  //    access for callers that hard-code `out[0] == delta` etc.
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
