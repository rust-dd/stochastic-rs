use ndarray::Array1;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::process::bm::Bm;
use stochastic_rs_stochastic::traits::ProcessExt;

use crate::GridPlotter;
use crate::plot_distribution;
use crate::plot_process;
use crate::plot_vol_surface;

#[test]
fn plot_process_writes_html() {
  let bm = Bm::new(64, Some(1.0), Unseeded);
  let path_arr = bm.sample();
  let mut out = std::env::temp_dir();
  out.push("stochastic_rs_test_plot_process.html");
  plot_process(&path_arr, out.to_str().unwrap());
  assert!(out.exists(), "plot_process did not write file");
  let _ = std::fs::remove_file(out);
}

#[test]
fn plot_distribution_writes_html() {
  let samples: Array1<f64> = (0..256).map(|i| (i as f64) * 0.01).collect();
  let mut out = std::env::temp_dir();
  out.push("stochastic_rs_test_plot_distribution.html");
  plot_distribution(&samples, out.to_str().unwrap(), "test");
  assert!(out.exists(), "plot_distribution did not write file");
  let _ = std::fs::remove_file(out);
}

#[test]
fn plot_vol_surface_writes_html() {
  use ndarray::Array2;
  let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
  let maturities = vec![0.25, 0.5, 1.0];
  let ivs = Array2::<f64>::from_shape_fn((maturities.len(), strikes.len()), |(j, i)| {
    0.2 + 0.01 * (j as f64) - 0.005 * ((i as f64) - 2.0).abs()
  });
  let mut out = std::env::temp_dir();
  out.push("stochastic_rs_test_plot_vol_surface.html");
  plot_vol_surface(&strikes, &maturities, &ivs, out.to_str().unwrap());
  assert!(out.exists(), "plot_vol_surface did not write file");
  let _ = std::fs::remove_file(out);
}

#[test]
#[should_panic(expected = "ivs shape must be")]
fn plot_vol_surface_rejects_bad_shape() {
  use ndarray::Array2;
  let strikes = vec![80.0, 90.0, 100.0];
  let maturities = vec![0.25, 0.5];
  let bad = Array2::<f64>::zeros((3, 5));
  plot_vol_surface(&strikes, &maturities, &bad, "/tmp/should_not_exist.html");
}

#[test]
fn grid_plotter_rescale_threshold_disabled_writes_html() {
  let bm = Bm::new(64, Some(1.0), Unseeded);
  let grid = GridPlotter::new()
    .title("rescale-threshold=None smoke")
    .cols(1)
    .rescale_threshold(None)
    .register(&bm, "Bm", 1);
  let plot = grid.plot();
  let mut out = std::env::temp_dir();
  out.push("stochastic_rs_test_rescale_disabled.html");
  plot.write_html(&out);
  assert!(out.exists(), "plot did not write file");
  let _ = std::fs::remove_file(out);
}
