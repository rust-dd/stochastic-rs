//! Benchmark: M-T vs FD for digital put 2D (closed-form g kernel).
//!
//! Requires the `openblas` feature. Run with:
//! `cargo test --release --features openblas mt_vs_fd_benchmark -- --nocapture --test-threads=1`

#![cfg(feature = "openblas")]

use std::time::Instant;

use ndarray::Array2;
use stochastic_rs::quant::pricing::malliavin_thalmaier::engine::MtGreeks;
use stochastic_rs::quant::pricing::malliavin_thalmaier::engine::MtPayoff;
use stochastic_rs::quant::pricing::malliavin_thalmaier::heston::AssetParams;
use stochastic_rs::quant::pricing::malliavin_thalmaier::heston::MultiHestonParams;

fn make_params_2d() -> MultiHestonParams<f64> {
  let a1 = AssetParams {
    s0: 100.0,
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    xi: 0.3,
    rho: -0.7,
  };
  let a2 = AssetParams {
    s0: 100.0,
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    xi: 0.3,
    rho: -0.7,
  };
  let mut cross = Array2::<f64>::eye(2);
  cross[[0, 1]] = 0.5;
  cross[[1, 0]] = 0.5;
  MultiHestonParams {
    assets: vec![a1, a2],
    cross_corr: cross,
    r: 0.05,
    tau: 1.0,
    n_steps: 252,
  }
}

fn fd_all_deltas(
  params: &MultiHestonParams<f64>,
  payoff: &MtPayoff<f64>,
  n_paths: usize,
) -> (Vec<f64>, f64) {
  let d = params.n_assets();
  let bump = 0.5;
  let mut deltas = vec![0.0; d];

  let t0 = Instant::now();
  for p in 0..d {
    let mut up = params.clone();
    up.assets[p].s0 += bump;
    let mut dn = params.clone();
    dn.assets[p].s0 -= bump;

    let price_up = MtGreeks::new(up, 0.01, n_paths).price(payoff);
    let price_dn = MtGreeks::new(dn, 0.01, n_paths).price(payoff);
    deltas[p] = (price_up - price_dn) / (2.0 * bump);
  }
  let elapsed = t0.elapsed().as_secs_f64();
  (deltas, elapsed)
}

/// Digital put 2D: M-T with closed-form g kernel (arctan + ln) vs FD.
#[test]
#[ignore = "benchmark: 50k path MC; run with: cargo test --release --features openblas mt_vs_fd_digital_put_2d -- --ignored --nocapture"]
fn mt_vs_fd_digital_put_2d() {
  let n_paths = 50_000;
  let params = make_params_2d();
  let payoff = MtPayoff::DigitalPut2D {
    strikes: [100.0, 100.0],
  };

  // FD: 4 MC runs (bump each asset up/down)
  let (fd_deltas, fd_time) = fd_all_deltas(&params, &payoff, n_paths);

  // M-T: 1 MC run, closed-form g kernel
  let engine = MtGreeks::new(params.clone(), 0.01, n_paths);
  let t0 = Instant::now();
  let mt_deltas = engine.all_deltas(&payoff);
  let mt_time = t0.elapsed().as_secs_f64();

  let speedup = fd_time / mt_time;

  println!("\n=== Digital Put 2D, {n_paths} paths ===");
  println!(
    "FD Δ₁={:.6}  Δ₂={:.6}  ({:.3}s, 4 MC runs)",
    fd_deltas[0], fd_deltas[1], fd_time
  );
  println!(
    "MT Δ₁={:.6}  Δ₂={:.6}  ({:.3}s, 1 MC run)",
    mt_deltas[0], mt_deltas[1], mt_time
  );
  println!("Speedup: {speedup:.1}x");

  // Both should be negative (higher spot → less likely to finish below strike)
  println!("\nSign check:");
  println!(
    "  FD: Δ₁<0? {}  Δ₂<0? {}",
    fd_deltas[0] < 0.0,
    fd_deltas[1] < 0.0
  );
  println!(
    "  MT: Δ₁<0? {}  Δ₂<0? {}",
    mt_deltas[0] < 0.0,
    mt_deltas[1] < 0.0
  );

  // By symmetry (same params), both deltas should be similar
  println!("\nSymmetry check (same params → Δ₁ ≈ Δ₂):");
  println!("  FD: |Δ₁-Δ₂| = {:.6}", (fd_deltas[0] - fd_deltas[1]).abs());
  println!("  MT: |Δ₁-Δ₂| = {:.6}", (mt_deltas[0] - mt_deltas[1]).abs());
}

/// Price comparison: M-T MC price vs analytical bounds.
#[test]
#[ignore = "benchmark: 50k path MC; run with: cargo test --release --features openblas mt_digital_put_2d_price_sanity -- --ignored --nocapture"]
fn mt_digital_put_2d_price_sanity() {
  let n_paths = 50_000;
  let params = make_params_2d();
  let payoff = MtPayoff::DigitalPut2D {
    strikes: [100.0, 100.0],
  };

  let engine = MtGreeks::new(params, 0.01, n_paths);
  let price = engine.price(&payoff);

  // ATM digital put on 2 correlated assets: price should be in (0, e^{-rT})
  // Roughly: P(S1<K1 AND S2<K2) * e^{-rT}
  // With positive drift and correlation, expect ~0.15-0.35
  let discount = (-0.05_f64).exp();
  println!("\n=== Digital Put 2D Price ===");
  println!("MC price = {price:.6}");
  println!("Discount = {discount:.6}");
  println!("Implied P(ITM) = {:.4}", price / discount);

  assert!(
    price > 0.0 && price < discount,
    "price {price} out of (0, {discount})"
  );
}
