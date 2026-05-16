//! Bit-exact validation: deterministic Fgn must produce identical eigenvalues
//! and sample paths regardless of internal construction changes.
//!
//! ```sh
//! cargo run --release --example fgn_bitexact
//! ```
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::noise::fgn::Fgn;
use stochastic_rs::traits::ProcessExt;

fn main() {
  let hursts = [0.2, 0.3, 0.5, 0.7, 0.9];
  let ns = [64, 256, 1024, 4096];
  let seeds = [1u64, 42, 12345];

  println!("Bit-exact eigenvalue reproducibility test");
  println!("=========================================\n");

  for &h in &hursts {
    for &n in &ns {
      let fgn_a = Fgn::<f64, _>::new(h, n, Some(1.0), Unseeded);
      let fgn_b = Fgn::<f64, _>::new(h, n, Some(1.0), Unseeded);

      assert_eq!(fgn_a.sqrt_eigenvalues.len(), fgn_b.sqrt_eigenvalues.len());
      let max_diff: f64 = fgn_a
        .sqrt_eigenvalues
        .iter()
        .zip(fgn_b.sqrt_eigenvalues.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

      assert_eq!(max_diff, 0.0, "eigenvalues differ for H={h}, n={n}");
      println!(
        "  H={h:.1}, n={n:>5}: eigenvalues bit-exact (len={})",
        fgn_a.sqrt_eigenvalues.len()
      );
    }
  }

  println!("\nDeterministic sample path reproducibility test");
  println!("==============================================\n");

  for &h in &hursts {
    for &n in &ns {
      for &seed in &seeds {
        let fgn_a = Fgn::new(h, n, Some(1.0), Deterministic::new(seed));
        let fgn_b = Fgn::new(h, n, Some(1.0), Deterministic::new(seed));

        let path_a = fgn_a.sample();
        let path_b = fgn_b.sample();

        assert_eq!(path_a.len(), path_b.len());
        let max_diff: f64 = path_a
          .iter()
          .zip(path_b.iter())
          .map(|(a, b)| (a - b).abs())
          .fold(0.0_f64, f64::max);

        assert_eq!(max_diff, 0.0, "paths differ for H={h}, n={n}, seed={seed}");
      }
      println!("  H={h:.1}, n={n:>5}: all seeds bit-exact");
    }
  }

  println!("\nScale and dt consistency test");
  println!("=============================\n");

  for &h in &hursts {
    for &n in &ns {
      for t in [0.5, 1.0, 2.5] {
        let fgn = Fgn::<f64, _>::new(h, n, Some(t), Unseeded);
        let expected_dt = t / n as f64;
        assert!(
          (fgn.dt() - expected_dt).abs() < 1e-15,
          "dt mismatch: H={h}, n={n}, t={t}"
        );
      }
    }
    println!("  H={h:.1}: dt and scale correct for all (n, t) pairs");
  }

  println!("\nAll tests passed.");
}
