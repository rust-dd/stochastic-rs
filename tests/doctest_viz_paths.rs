// docs: viz#sample-path-overlay
//! Backs the sample-path overlay example on the visualization page.
//! `stochastic-rs-viz` is an unconditional umbrella dependency (there is
//! no `viz` cargo feature) and is re-exported as `visualization`, not
//! `viz`.

use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::diffusion::ou::Ou;
use stochastic_rs::visualization::GridPlotter;

#[test]
fn grid_plotter_registers_process_paths() {
  let p = Ou::<f64, _>::new(
    2.0,
    0.0,
    1.0,
    1_000,
    Some(0.0),
    Some(1.0),
    Deterministic::new(7),
  );

  let plot = GridPlotter::new().register(&p, "OU", 8);
  // A temp path keeps this example from writing into the working tree.
  let path = std::env::temp_dir().join("stochastic_rs_doctest_ou_paths.html");
  plot.write_html(&path);
  assert!(path.exists());
}
