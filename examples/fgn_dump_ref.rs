//! Dumps Fgn reference data (eigenvalues + deterministic paths) to npy files.
use ndarray_npy::write_npy;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::noise::fgn::Fgn;
use stochastic_rs::traits::ProcessExt;

fn main() {
  let cases: &[(f64, usize)] = &[(0.3, 256), (0.5, 1024), (0.7, 4096)];
  let seed = 42u64;

  std::fs::create_dir_all("target/fgn_ref").unwrap();

  for &(h, n) in cases {
    let fgn = Fgn::new(h, n, Some(1.0), Deterministic::new(seed));
    let eig = fgn.sqrt_eigenvalues.as_ref().clone();
    let path = fgn.sample();

    let tag = format!("h{}_n{}", (h * 100.0) as u32, n);
    write_npy(format!("target/fgn_ref/eig_{tag}.npy"), &eig).unwrap();
    write_npy(format!("target/fgn_ref/path_{tag}.npy"), &path).unwrap();
    println!("wrote {tag}: eig[{}] path[{}]", eig.len(), path.len());
  }
}
