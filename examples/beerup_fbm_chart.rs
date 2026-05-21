//! Generates real fBm paths for H = 0.15, 0.5, 0.85 and dumps them as
//! `t value` .dat files in docs/ for inclusion in the beamer pgfplots.
use std::fs::File;
use std::io::Write;

use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::noise::fgn::Fgn;
use stochastic_rs::traits::ProcessExt;

fn main() {
  let out_dir = "docs";
  std::fs::create_dir_all(out_dir).unwrap();

  let cases: &[(f64, u64, &str)] = &[
    (0.15, 7, "fbm_h15"),
    (0.50, 11, "fbm_h50"),
    (0.85, 3, "fbm_h85"),
  ];
  let n: usize = 1024;

  for &(h, seed, tag) in cases {
    let fgn = Fgn::<f64, _>::new(h, n, Some(1.0), Deterministic::new(seed));
    let inc = fgn.sample();
    let mut fbm = Vec::with_capacity(n + 1);
    fbm.push(0.0f64);
    for v in inc.iter() {
      fbm.push(fbm.last().unwrap() + v);
    }
    let max_abs = fbm
      .iter()
      .map(|x| x.abs())
      .fold(0.0f64, f64::max)
      .max(1e-12);
    let scale = 1.8 / max_abs;

    let path_file = format!("{out_dir}/{tag}.dat");
    let mut f = File::create(&path_file).unwrap();
    writeln!(f, "t v").unwrap();
    for (i, v) in fbm.iter().enumerate() {
      let t = i as f64 / n as f64;
      writeln!(f, "{:.6} {:.6}", t, v * scale).unwrap();
    }
    println!(
      "wrote {path_file} ({} samples, H={h}, seed={seed})",
      fbm.len()
    );
  }
}
