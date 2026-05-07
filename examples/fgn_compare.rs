//! Compares old vs new Fgn reference data from npy files.
use ndarray::Array1;
use ndarray_npy::read_npy;

fn main() {
  let cases = [("h30_n256", 0.3), ("h50_n1024", 0.5), ("h70_n4096", 0.7)];

  println!("Old vs New Fgn comparison");
  println!("========================\n");

  for (tag, h) in &cases {
    let eig_old: Array1<f64> = read_npy(format!("target/fgn_ref_old/eig_{tag}.npy")).unwrap();
    let eig_new: Array1<f64> = read_npy(format!("target/fgn_ref/eig_{tag}.npy")).unwrap();
    let path_old: Array1<f64> = read_npy(format!("target/fgn_ref_old/path_{tag}.npy")).unwrap();
    let path_new: Array1<f64> = read_npy(format!("target/fgn_ref/path_{tag}.npy")).unwrap();

    assert_eq!(eig_old.len(), eig_new.len());
    assert_eq!(path_old.len(), path_new.len());

    let eig_max_abs = eig_old
      .iter()
      .zip(eig_new.iter())
      .map(|(a, b)| (a - b).abs())
      .fold(0.0_f64, f64::max);
    let eig_max_rel = eig_old
      .iter()
      .zip(eig_new.iter())
      .filter(|(a, _)| a.abs() > 1e-15)
      .map(|(a, b)| ((a - b) / a).abs())
      .fold(0.0_f64, f64::max);
    let eig_identical = eig_old
      .iter()
      .zip(eig_new.iter())
      .all(|(a, b)| a.to_bits() == b.to_bits());

    let path_max_abs = path_old
      .iter()
      .zip(path_new.iter())
      .map(|(a, b)| (a - b).abs())
      .fold(0.0_f64, f64::max);
    let path_identical = path_old
      .iter()
      .zip(path_new.iter())
      .all(|(a, b)| a.to_bits() == b.to_bits());

    println!("H={h} ({tag}):");
    println!(
      "  eigenvalues: bit-exact={eig_identical}, max_abs_diff={eig_max_abs:.2e}, max_rel_diff={eig_max_rel:.2e}"
    );
    println!("  paths:       bit-exact={path_identical}, max_abs_diff={path_max_abs:.2e}");

    println!("\n  eigenvalues (first 8):");
    println!(
      "  {:>4}  {:>22}  {:>22}  {:>10}",
      "idx", "old", "new", "diff"
    );
    for i in 0..8.min(eig_old.len()) {
      println!(
        "  {:>4}  {:>22.16e}  {:>22.16e}  {:>10.2e}",
        i,
        eig_old[i],
        eig_new[i],
        (eig_old[i] - eig_new[i]).abs()
      );
    }

    println!("\n  paths (first 10):");
    println!(
      "  {:>4}  {:>22}  {:>22}  {:>10}",
      "idx", "old", "new", "diff"
    );
    for i in 0..10.min(path_old.len()) {
      println!(
        "  {:>4}  {:>22.16e}  {:>22.16e}  {:>10.2e}",
        i,
        path_old[i],
        path_new[i],
        (path_old[i] - path_new[i]).abs()
      );
    }
    println!();
  }
}
