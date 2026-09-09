//! The numbers behind the landing page's figures.
//!
//! Not part of the workspace: copy it to `examples/figure_data.rs` at the
//! repository root and run
//!
//! ```text
//! cargo run --release --example figure_data
//! ```
//!
//! then feed the output through the polyline conversion in
//! `website/lib/figures.ts`'s header comment. It lives here rather than in
//! `examples/` because it is a documentation asset with no consumer in the
//! crate, and an example that nothing runs rots.

use stochastic_rs::prelude::*;
use stochastic_rs::quant::vol_surface::svi::SviRawParams;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::process::fbm::Fbm;
use stochastic_rs::stochastic::volatility::HestonPow;
use stochastic_rs::stochastic::volatility::heston::Heston;

/// A path thinned to `k` points and rounded, so the SVG stays small.
fn thin(path: &[f64], k: usize, digits: i32) -> Vec<f64> {
  let step = (path.len() - 1) as f64 / (k - 1) as f64;
  let f = 10f64.powi(digits);
  (0..k)
    .map(|i| (path[(i as f64 * step).round() as usize] * f).round() / f)
    .collect()
}

fn main() {
  let heston = Heston::<f64, _>::new(
    Some(100.0),
    Some(0.04),
    2.0,
    0.04,
    0.3,
    -0.7,
    0.03,
    512,
    Some(1.0),
    HestonPow::Sqrt,
    Some(false),
    Deterministic::new(21),
  );
  println!("HESTON");
  for p in heston.sample_par(14) {
    println!("{:?}", thin(p[0].as_slice().unwrap(), 65, 2));
  }

  // The same seed at three Hurst exponents, so only the roughness differs.
  println!("FBM");
  for h in [0.3, 0.5, 0.8] {
    let path = Fbm::<f64, _>::new(h, 512, Some(1.0), Deterministic::new(4)).sample();
    println!("{h} {:?}", thin(path.as_slice().unwrap(), 65, 3));
  }

  println!("SVI");
  for (tau, a, b, rho, m, sig) in [
    (0.25, 0.008, 0.10, -0.45, 0.02, 0.10),
    (0.50, 0.016, 0.13, -0.40, 0.02, 0.13),
    (1.00, 0.030, 0.17, -0.35, 0.03, 0.17),
    (2.00, 0.058, 0.22, -0.30, 0.04, 0.22),
  ] {
    let svi = SviRawParams::new(a, b, rho, m, sig);
    let vols: Vec<f64> = (0..41)
      .map(|i| {
        let k = -0.5 + i as f64 * 0.025;
        (svi.implied_vol(k, tau) * 10_000.0).round() / 10_000.0
      })
      .collect();
    println!("{tau} {vols:?}");
  }
}
