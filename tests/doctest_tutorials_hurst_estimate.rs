// docs: tutorials/hurst-exponent
//! Estimates the Hurst exponent of a simulated fBm path with four estimators behind one trait.

use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stats::fractal_dim::Higuchi;
use stochastic_rs::stats::hurst::Dfa;
use stochastic_rs::stats::hurst::Gph;
use stochastic_rs::stats::hurst::Wavelet;
use stochastic_rs::stochastic::process::fbm::Fbm;
use stochastic_rs::traits::HurstEstimator;
use stochastic_rs::traits::ProcessExt;

#[test]
fn estimate_hurst_of_an_fbm_path() {
  let path = Fbm::<f64, _>::new(0.7, 4_096, Some(1.0), Deterministic::new(1)).sample();

  // Every default here takes the level path, not its increments.
  let dfa = Dfa::default().estimate(path.view()).unwrap();
  let gph = Gph::default().estimate(path.view()).unwrap();
  let wavelet = Wavelet::default().estimate(path.view()).unwrap();
  let higuchi = Higuchi::default().estimate(path.view()).unwrap(); // H = 2 - D

  for h in [dfa.hurst, gph.hurst, wavelet.hurst, higuchi.hurst] {
    assert!((h - 0.7).abs() < 0.1, "H = {h}");
  }
  // Only Gph and Wavelet report an asymptotic standard error.
  assert!(gph.std_err.is_some() && wavelet.std_err.is_some() && dfa.std_err.is_none());
}
