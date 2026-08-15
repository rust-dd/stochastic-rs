//! Plot a stochastic process: five GBM sample paths overlaid on one chart.
//!
//! Writes a Plotly HTML chart to `target/plot_process_gbm.html`. Run with
//! `cargo run --release --example plot_process`.
use plotly::Layout;
use plotly::Plot;
use plotly::Scatter;
use plotly::common::Line;
use plotly::common::Mode;
use plotly::layout::Axis;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::diffusion::gbm::Gbm;
use stochastic_rs::traits::ProcessExt;

fn main() {
  let n = 252; // one trading year of daily steps
  let mu = 0.05;
  let sigma = 0.2;
  let x0 = 100.0;
  let n_paths = 5_u64;

  let xs = (0..n)
    .map(|i| i as f64 / (n - 1) as f64)
    .collect::<Vec<f64>>();
  let mut plot = Plot::new();

  for seed in 1..=n_paths {
    let gbm = Gbm::<f64, _>::new(mu, sigma, n, Some(x0), Some(1.0), Deterministic::new(seed));
    let path = gbm.sample();
    plot.add_trace(
      Scatter::new(xs.clone(), path.to_vec())
        .name(format!("path {seed}"))
        .mode(Mode::Lines)
        .line(Line::new().width(1.4)),
    );
  }

  let layout = Layout::new()
    .title(format!("GBM sample paths (μ={mu}, σ={sigma}, S₀={x0})"))
    .x_axis(Axis::new().title("t"))
    .y_axis(Axis::new().title("S_t"));
  plot.set_layout(layout);

  let out_path = "target/plot_process_gbm.html";
  plot.write_html(out_path);
  println!("Wrote {out_path}");
}
