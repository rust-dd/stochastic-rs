// docs: viz#implied-vol-surface-heatmap
//! Backs the implied-vol surface heatmap example on the visualization
//! page. Uses the `plot_vol_surface` convenience function directly —
//! `ImpliedVolSurface` has no `from_market` constructor. `visualization`
//! is the umbrella's re-export name for `stochastic-rs-viz` (not `viz`).

use ndarray::array;
use stochastic_rs::visualization::plot_vol_surface;

#[test]
fn plot_vol_surface_writes_html() {
  let strikes = [90.0, 100.0, 110.0];
  let maturities = [0.5, 1.0];
  let ivs = array![[0.24, 0.20, 0.23], [0.22, 0.19, 0.21]]; // shape (N_T, N_K)

  let path = std::env::temp_dir().join("stochastic_rs_doctest_iv_surface.html");
  plot_vol_surface(&strikes, &maturities, &ivs, path.to_str().unwrap());
  assert!(path.exists());
}
