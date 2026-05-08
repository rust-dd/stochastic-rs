//! Convenience one-shot plotters. Each writes a self-contained HTML file
//! and is meant for quick visualization rather than for use in production
//! reports — for the latter, use [`crate::GridPlotter`].

use ndarray::Array1;
use ndarray::Array2;
use plotly::Plot;
use plotly::Scatter;
use plotly::common::Mode;
use stochastic_rs_distributions::traits::FloatExt;

/// Convenience: plot a single 1D process sample as HTML.
///
/// ```ignore
/// use stochastic_rs_stochastic::process::bm::Bm;
/// use stochastic_rs_stochastic::traits::ProcessExt;
/// use stochastic_rs_viz::plot_process;
///
/// let bm = Bm::new(1000, Some(1.0));
/// plot_process(&bm.sample(), "bm.html");
/// ```
pub fn plot_process<T: FloatExt>(sample: &Array1<T>, path: &str) {
  let xs: Vec<f64> = (0..sample.len()).map(|i| i as f64).collect();
  let ys: Vec<f64> = sample.iter().map(|v| v.to_f64().unwrap()).collect();
  let mut plot = Plot::new();
  plot.add_trace(Scatter::new(xs, ys).mode(Mode::Lines).name("path"));
  plot.write_html(path);
}

/// Convenience: plot a histogram of distribution samples to an HTML file.
pub fn plot_distribution<T: FloatExt>(samples: &Array1<T>, path: &str, title: &str) {
  use plotly::Histogram;
  let mut plot = Plot::new();
  let xs: Vec<f64> = samples.iter().map(|v| v.to_f64().unwrap()).collect();
  plot.add_trace(Histogram::new(xs).name(title));
  plot.write_html(path);
}

/// Convenience: plot a 3D vol-surface (strikes × maturities × IV) to an HTML file.
pub fn plot_vol_surface(strikes: &[f64], maturities: &[f64], ivs: &Array2<f64>, path: &str) {
  use plotly::Surface;
  assert_eq!(
    ivs.dim(),
    (maturities.len(), strikes.len()),
    "ivs shape must be (N_T, N_K)"
  );
  let z: Vec<Vec<f64>> = (0..maturities.len())
    .map(|j| (0..strikes.len()).map(|i| ivs[[j, i]]).collect())
    .collect();
  let mut plot = Plot::new();
  plot.add_trace(Surface::new(z).x(strikes.to_vec()).y(maturities.to_vec()));
  plot.write_html(path);
}
