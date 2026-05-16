//! Debug: print the Whittle likelihood surface for known fBM.
//!
//! This is a manual diagnostic, not a regression test — it prints the
//! periodogram and NLL surface for visual inspection. Run explicitly via
//! `cargo test -- --ignored likelihood_surface` when investigating Fukasawa
//! behaviour.

use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stats::fukasawa_hurst;
use stochastic_rs::stochastic::process::fbm::Fbm;
use stochastic_rs::traits::ProcessExt;

#[test]
#[ignore = "diagnostic-only: prints likelihood surface, no assertions"]
fn likelihood_surface() {
  let true_h = 0.3_f64;
  let n = 1024;
  let fbm = Fbm::new(true_h, n, Some(1.0), Unseeded);
  let path = fbm.sample();

  // fGN increments → log squared increments (as "log RV")
  let log_rv: Vec<f64> = (1..n)
    .map(|i| {
      let r: f64 = path[i] - path[i - 1];
      (r * r).max(1e-20).ln()
    })
    .collect();

  // Y = increments of log_rv
  let y: Vec<f64> = (1..log_rv.len())
    .map(|i| log_rv[i] - log_rv[i - 1])
    .collect();
  let ny = y.len();
  let n_freq = ny / 2;

  // Periodogram
  let pgram: Vec<f64> = (1..=n_freq)
    .map(|j| {
      let lam = 2.0 * std::f64::consts::PI * j as f64 / ny as f64;
      let mut cr = 0.0;
      let mut ci = 0.0;
      for (t, &yt) in y.iter().enumerate() {
        cr += yt * (lam * t as f64).cos();
        ci += yt * (lam * t as f64).sin();
      }
      (cr * cr + ci * ci) / (2.0 * std::f64::consts::PI * ny as f64)
    })
    .collect();

  println!(
    "\nPeriodogram stats: n={}, mean={:.6}, max={:.6}",
    pgram.len(),
    pgram.iter().sum::<f64>() / pgram.len() as f64,
    pgram.iter().cloned().fold(0.0_f64, f64::max)
  );

  // Print spectral density at a few freqs for different H
  println!("\nSpectral density g(λ, H, v=1.0) at λ=0.1:");
  for h_idx in &[5, 10, 20, 30, 40, 49] {
    let h = *h_idx as f64 * 0.01;
    let g = fukasawa_hurst::spectral_density(0.1, h, 1.0, 1, ny, 500);
    println!("  H={:.2} → g={:.6}", h, g);
  }

  // Print NLL surface
  println!("\nNLL surface U(H, v) — true H={true_h}:");
  println!(
    "{:<6} {:<8} {:<8} {:<8} {:<8} {:<8}",
    "H", "v=0.25", "v=0.5", "v=1.0", "v=2.0", "v=5.0"
  );
  for h_idx in &[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49] {
    let h = *h_idx as f64 * 0.01;
    let mut vals = Vec::new();
    for &v in &[0.25, 0.5, 1.0, 2.0, 5.0] {
      let nll = fukasawa_hurst::whittle_objective(&pgram, h, v, 1, ny, 1e-5, 500);
      vals.push(format!("{:.4}", nll));
    }
    println!(
      "{:<6.2} {:<8} {:<8} {:<8} {:<8} {:<8}",
      h, vals[0], vals[1], vals[2], vals[3], vals[4]
    );
  }
}
