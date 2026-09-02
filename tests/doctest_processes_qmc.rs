// docs: processes#quasi-monte-carlo-paths-sobol--brownian-bridge
//! Backs the quasi-Monte Carlo example on the processes page.

use ndarray::Array2;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::mc::brownian_bridge_qmc::BrownianBridgeQmc;
use stochastic_rs::stochastic::mc::sobol::SobolSeq;

#[test]
fn qmc_brownian_paths_price_a_martingale() {
  // 2^12 scrambled Sobol paths of a one-year Brownian motion on 64 steps.
  let qmc = BrownianBridgeQmc::scrambled(64, 1.0, &Deterministic::new(7));
  let w: Array2<f64> = qmc.paths(4_096);
  assert_eq!(w.dim(), (4_096, 64));

  // exp(σW_T − σ²T/2) is a martingale, so its QMC average sits at 1 to a
  // few parts in a thousand — tighter than plain MC with the same budget.
  let sigma = 0.2;
  let mean = w
    .column(63)
    .iter()
    .map(|x| (sigma * x - 0.5 * sigma * sigma).exp())
    .sum::<f64>()
    / 4_096.0;
  assert!((mean - 1.0).abs() < 5e-3, "martingale mean {mean}");

  // The sequence itself reaches far beyond the old 21-dimension ceiling.
  let deep: Array2<f64> = SobolSeq::new(5_000).sample(16);
  assert!(deep.iter().all(|u| (0.0..1.0).contains(u)));
}
