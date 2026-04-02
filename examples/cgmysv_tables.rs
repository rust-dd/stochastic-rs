//! Reproduces the numerical results from Kim (2021), arXiv:2101.11001.
//!
//! Tables 4, 8, and 9.

use stochastic_rs::quant::pricing::barrier::BarrierType;
use stochastic_rs::quant::pricing::cgmysv::{CgmysvModel, CgmysvParams, CgmysvPricer};
use stochastic_rs::quant::pricing::fourier::{CGMYFourier, CarrMadanPricer, GilPelaezPricer, LewisPricer};
use stochastic_rs::quant::OptionType;

fn main() {
  // Table 2: Calibrated parameters (call side)
  let call_params = CgmysvParams {
    alpha: 0.5184,
    lambda_plus: 25.4592,
    lambda_minus: 4.6040,
    kappa: 1.0029,
    eta: 0.0711,
    zeta: 0.3443,
    rho: -2.0283,
    v0: 0.006381,
  };

  let s0 = 2488.11;
  let r = 0.01213;
  let q = 0.01884;
  let k = 2500.0_f64;
  let tau28 = 28.0 / 365.0;

  println!("══════════════════════════════════════════════════════════════");
  println!("  Table 4 — European SPX (K=2500, T=28 days)");
  println!("══════════════════════════════════════════════════════════════");

  // FFT price
  let model = CgmysvModel {
    params: call_params.clone(),
    r,
    q,
  };
  let cm = CarrMadanPricer::default();
  let (log_k, calls) = cm.price_call_surface(&model, s0, r, tau28);
  let target_lnk = k.ln();
  let idx = log_k
    .iter()
    .enumerate()
    .min_by(|(_, a), (_, b)| ((**a) - target_lnk).abs().total_cmp(&((**b) - target_lnk).abs()))
    .unwrap()
    .0;
  let fft_call = calls[idx];
  let gp_call = GilPelaezPricer::price_call(&model, s0, k, r, q, tau28);
  let lw_call = LewisPricer::price_call(&model, s0, k, r, q, tau28);
  let cm_interp = cm.price_call(&model, s0, k, r, tau28);
  println!("  CarrMadan grid:  {fft_call:.4}   (nearest grid point)");
  println!("  CarrMadan interp:{cm_interp:.4}");
  println!("  Gil-Pelaez:      {gp_call:.4}");
  println!("  Lewis:           {lw_call:.4}");
  println!("  Paper FFT:       19.6590");

  // MCS prices
  for &n_paths in &[100, 1000, 5000, 10000] {
    let pricer = CgmysvPricer {
      params: call_params.clone(),
      s: s0,
      r,
      q,
      n_paths,
      n_steps: 100,
      n_jumps: 1024,
    };
    let call = pricer.price_european(k, tau28, OptionType::Call);
    let put = pricer.price_european(k, tau28, OptionType::Put);
    println!(
      "  MCS({n_paths:>5}) call: {:<20} put: {}",
      format!("{call}"),
      put
    );
  }
  // Try v0 = 0.06381 (possible Table 2 misread)
  let alt_params = CgmysvParams { v0: 0.06381, ..call_params.clone() };
  let alt_model = CgmysvModel { params: alt_params, r, q };
  let (log_k_alt, calls_alt) = cm.price_call_surface(&alt_model, s0, r, tau28);
  let idx_alt = log_k_alt.iter().enumerate()
    .min_by(|(_, a), (_, b)| ((**a) - target_lnk).abs().total_cmp(&((**b) - target_lnk).abs())).unwrap().0;
  println!("  FFT (v0=0.06381): {:.4}", calls_alt[idx_alt]);
  println!("  Paper MCS(10000) call: 19.6840 ± 0.2551   put: 32.6914 ± 0.7617");
  println!("  Paper FFT        call: 19.6590             put: 32.9541");
  println!("  Market           call: 19.05               put: 31.50");

  // Diagnostic: equivalent CGMY (no stochastic vol) for comparison
  // For constant v = v₀ over short T: effective c ≈ v₀ · C
  let big_c = call_params.norm_const();
  let c_eff = big_c * call_params.v0;
  println!("  C_norm = {big_c:.4}, c_eff(v0) = {c_eff:.6}, c_eff(eta) = {:.6}", big_c * call_params.eta);
  let cgmy_model = CGMYFourier {
    c: c_eff,
    g: call_params.lambda_minus, // G (positive jump tempering) = λ₋ in Kim
    m: call_params.lambda_plus,  // M (negative jump tempering) = λ₊ in Kim
    y: call_params.alpha,
    r,
    q,
  };
  let (log_k2, calls2) = cm.price_call_surface(&cgmy_model, s0, r, tau28);
  let idx2 = log_k2
    .iter()
    .enumerate()
    .min_by(|(_, a), (_, b)| ((**a) - target_lnk).abs().total_cmp(&((**b) - target_lnk).abs()))
    .unwrap()
    .0;
  println!(
    "  CGMY (v=η, ρ=0) call: {:.4}   (c_eff={c_eff:.4}, g={:.4}, m={:.4})",
    calls2[idx2], call_params.lambda_minus, call_params.lambda_plus
  );
  // Also try swapped g/m mapping
  let cgmy_swapped = CGMYFourier {
    c: c_eff,
    g: call_params.lambda_plus,
    m: call_params.lambda_minus,
    y: call_params.alpha,
    r,
    q,
  };
  let (log_k3, calls3) = cm.price_call_surface(&cgmy_swapped, s0, r, tau28);
  let idx3 = log_k3
    .iter()
    .enumerate()
    .min_by(|(_, a), (_, b)| ((**a) - target_lnk).abs().total_cmp(&((**b) - target_lnk).abs()))
    .unwrap()
    .0;
  println!(
    "  CGMY (swapped)   call: {:.4}   (g={:.4}, m={:.4})",
    calls3[idx3], call_params.lambda_plus, call_params.lambda_minus
  );

  // Table 8 & 9: Asian and Barrier
  let exotic_params = CgmysvParams {
    alpha: 0.52,
    lambda_plus: 25.46,
    lambda_minus: 4.604,
    kappa: 1.003,
    eta: 0.0711,
    zeta: 0.3443,
    rho: -2.0280,
    v0: 0.0064,
  };
  let tau25 = 25.0 / 365.0;

  println!();
  println!("══════════════════════════════════════════════════════════════");
  println!("  Table 8 — Asian (K=2500, T=25 days)");
  println!("══════════════════════════════════════════════════════════════");

  for &n_paths in &[100, 1000, 5000, 10000] {
    let pricer = CgmysvPricer {
      params: exotic_params.clone(),
      s: 2488.0,
      r: 0.0121,
      q: 0.0188,
      n_paths,
      n_steps: 100,
      n_jumps: 1024,
    };
    let call = pricer.price_asian(k, tau25, OptionType::Call);
    let put = pricer.price_asian(k, tau25, OptionType::Put);
    println!(
      "  MCS({n_paths:>5}) call: {:<20} put: {}",
      format!("{call}"),
      put
    );
  }
  println!("  Paper MCS(10000) call: 21.6513 ± 0.1937   put: 9.9964 ± 0.3679");

  println!();
  println!("══════════════════════════════════════════════════════════════");
  println!("  Table 9 — Barrier (K=2500, T=25 days)");
  println!("══════════════════════════════════════════════════════════════");

  for &n_paths in &[100, 1000, 5000, 10000] {
    let pricer = CgmysvPricer {
      params: exotic_params.clone(),
      s: 2488.0,
      r: 0.0121,
      q: 0.0188,
      n_paths,
      n_steps: 100,
      n_jumps: 1024,
    };
    let do_call = pricer.price_barrier(k, tau25, 2400.0, BarrierType::DownAndOut, OptionType::Call);
    let uo_put = pricer.price_barrier(k, tau25, 2750.0, BarrierType::UpAndOut, OptionType::Put);
    println!(
      "  MCS({n_paths:>5}) DO call: {:<20} UO put: {}",
      format!("{do_call}"),
      uo_put
    );
  }
  println!("  Paper MCS(10000) DO call: 16.5518 ± 0.1749   UO put: 30.2590 ± 0.3026");
}
