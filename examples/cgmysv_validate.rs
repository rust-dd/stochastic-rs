use stochastic_rs::quant::pricing::barrier::BarrierType;
use stochastic_rs::quant::pricing::cgmysv::{CgmysvModel, CgmysvParams, CgmysvPricer};
use stochastic_rs::quant::pricing::fourier::{CarrMadanPricer, GilPelaezPricer, LewisPricer};
use stochastic_rs::quant::OptionType;

fn main() {
  // v0 = 0.01115 (corrected from PDF misread 0.006381)
  let call_params = CgmysvParams {
    alpha: 0.5184, lambda_plus: 25.4592, lambda_minus: 4.6040,
    kappa: 1.0029, eta: 0.0711, zeta: 0.3443, rho: -2.0283, v0: 0.01115,
  };
  let s0 = 2488.11;
  let r = 0.01213;
  let q = 0.01884;

  println!("══════════════════════════════════════════════════════════════");
  println!("  Table 4 — European SPX (K=2500, T=28 days)");
  println!("══════════════════════════════════════════════════════════════");
  let tau28 = 28.0 / 365.0;
  let k = 2500.0_f64;
  let model = CgmysvModel { params: call_params.clone(), r, q };

  let cm = CarrMadanPricer::default();
  let cm_call = cm.price_call(&model, s0, k, r, tau28);
  let gp_call = GilPelaezPricer::price_call(&model, s0, k, r, q, tau28);
  let lw_call = LewisPricer::price_call(&model, s0, k, r, q, tau28);
  println!("  CarrMadan:   {cm_call:.4}");
  println!("  Gil-Pelaez:  {gp_call:.4}");
  println!("  Lewis:       {lw_call:.4}");

  for &n_paths in &[1000, 5000, 10000] {
    let pricer = CgmysvPricer {
      params: call_params.clone(), s: s0, r, q,
      n_paths, n_steps: 100, n_jumps: 1024,
    };
    let call = pricer.price_european(k, tau28, OptionType::Call);
    let put = pricer.price_european(k, tau28, OptionType::Put);
    println!("  MCS({n_paths:>5}) call: {:<20} put: {}", format!("{call}"), put);
  }
  println!("  Paper       call: 19.6840 ± 0.2551   put: 32.6914 ± 0.7617");
  println!("  Paper FFT   call: 19.6590             put: 32.9541");

  // Table 8 — Asian
  println!();
  println!("══════════════════════════════════════════════════════════════");
  println!("  Table 8 — Asian (K=2500, T=25 days)");
  println!("══════════════════════════════════════════════════════════════");
  let exotic_params = CgmysvParams {
    alpha: 0.52, lambda_plus: 25.46, lambda_minus: 4.604,
    kappa: 1.003, eta: 0.0711, zeta: 0.3443, rho: -2.0280,
    v0: 0.0064, // Section 4.3 uses these params as-is
  };
  let tau25 = 25.0 / 365.0;
  // but v0 here might also be misread — try same correction ratio
  let exotic_v0_corrected = CgmysvParams { v0: 0.0111, ..exotic_params.clone() };

  for (label, params) in [("paper v0=0.0064", &exotic_params), ("corrected v0=0.0111", &exotic_v0_corrected)] {
    println!("  [{label}]");
    for &n_paths in &[5000, 10000] {
      let pricer = CgmysvPricer {
        params: params.clone(), s: 2488.0, r: 0.0121, q: 0.0188,
        n_paths, n_steps: 100, n_jumps: 1024,
      };
      let call = pricer.price_asian(k, tau25, OptionType::Call);
      let put = pricer.price_asian(k, tau25, OptionType::Put);
      println!("    MCS({n_paths:>5}) call: {:<20} put: {}", format!("{call}"), put);
    }
  }
  println!("  Paper       call: 21.6513 ± 0.1937   put: 9.9964 ± 0.3679");

  // Table 9 — Barrier
  println!();
  println!("══════════════════════════════════════════════════════════════");
  println!("  Table 9 — Barrier (K=2500, T=25 days)");
  println!("══════════════════════════════════════════════════════════════");
  for (label, params) in [("paper v0=0.0064", &exotic_params), ("corrected v0=0.0111", &exotic_v0_corrected)] {
    println!("  [{label}]");
    for &n_paths in &[5000, 10000] {
      let pricer = CgmysvPricer {
        params: params.clone(), s: 2488.0, r: 0.0121, q: 0.0188,
        n_paths, n_steps: 100, n_jumps: 1024,
      };
      let do_call = pricer.price_barrier(k, tau25, 2400.0, BarrierType::DownAndOut, OptionType::Call);
      let uo_put = pricer.price_barrier(k, tau25, 2750.0, BarrierType::UpAndOut, OptionType::Put);
      println!("    MCS({n_paths:>5}) DO call: {:<20} UO put: {}", format!("{do_call}"), uo_put);
    }
  }
  println!("  Paper       DO call: 16.5518 ± 0.1749   UO put: 30.2590 ± 0.3026");
}
